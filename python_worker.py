import asyncio
import json
import os

import aio_pika
import psycopg2
from aio_pika.abc import AbstractIncomingMessage

from gpu_worker import join_gpu_worker, start_gpu_worker, stop_gpu_worker
from processor import process_single_tender
from utils.config import RABBIT_URL

POSTGRES_DSN = os.getenv("POSTGRES_CONN_STRING")
MAX_ATTEMPTS = 5  # keep synced with your Go maxAttempts constant


def connect_db():
    # Consider replacing with a connection pool for higher throughput.
    return psycopg2.connect(POSTGRES_DSN)


def claim_job(conn, job_id):
    """
    Atomically claim a job. Returns tuple (payload_json, attempts) or None.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status='running_python', attempts = attempts + 1, updated_at=NOW()
            WHERE id=%s AND status='pending_python'
            RETURNING payload, attempts;
        """,
            (job_id,),
        )
        row = cur.fetchone()
        conn.commit()
        return row  # None if not claimable


def reset_job_to_pending(conn, job_id):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status='pending_python', updated_at=NOW()
            WHERE id=%s
        """,
            (job_id,),
        )
        conn.commit()


def complete_job(conn, job_id, python_result):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET python_result=%s::jsonb,
                status='completed',
                updated_at=NOW()
            WHERE id=%s;
        """,
            (json.dumps(python_result), job_id),
        )
        conn.commit()


def fail_job(conn, job_id, error_msg):
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE jobs
            SET status='failed',
                last_error=%s,
                updated_at=NOW()
            WHERE id=%s;
        """,
            (error_msg, job_id),
        )
        conn.commit()


async def on_message(message: AbstractIncomingMessage):
    """
    NOTE: we use message.process() context manager which will ACK the message if the block
    exits normally and will NACK (requeue) if an exception is raised. To get correct retry
    behaviour we handle DB state explicitly inside the block and re-raise when we want the
    broker to retry the message.
    """
    async with message.process():
        job_msg = json.loads(message.body)
        job_id = job_msg["job_id"]

        conn = connect_db()
        claimed = claim_job(conn, job_id)

        if not claimed:
            conn.close()
            # Nothing to do: job not pending_python, just ack and drop
            return

        payload_json, attempts = claimed

        # Payload from Postgres may be a string (JSON). Ensure we have a dict.
        if isinstance(payload_json, (bytes, str)):
            try:
                payload = json.loads(payload_json)
            except Exception:
                # invalid payload stored in DB: mark job failed and ack message
                fail_job(conn, job_id, "invalid payload JSON")
                conn.close()
                return
        else:
            payload = payload_json

        try:
            # process_single_tender can perform blocking work; ensure it runs without blocking the event loop
            python_result = await process_single_tender(payload)
            complete_job(conn, job_id, python_result)

            print(f"[python-worker] DONE job_id={job_id}")

            conn.close()
            return

        except Exception as e:
            # Decide whether to permanently fail or requeue for retry
            err_str = str(e)
            if attempts >= MAX_ATTEMPTS:
                # mark failed and ACK (by returning without raising)
                fail_job(conn, job_id, err_str)
                conn.close()
                return
            else:
                # reset job state to pending_python so the message will be claimable again
                # then re-raise to NACK and allow RabbitMQ to redeliver after its retry/backoff
                try:
                    reset_job_to_pending(conn, job_id)
                finally:
                    conn.close()
                # Re-raise so aio_pika will NACK the message and broker can redeliver
                raise


async def start_worker():
    # GPU thread MUST start before RabbitMQ listening
    gpu_thread = start_gpu_worker()

    print("Connecting to RabbitMQ...")
    connection = await aio_pika.connect_robust(RABBIT_URL)

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    queue = await channel.declare_queue("jobs.python", durable=True)

    print("Python async worker started. Waiting for messages...")
    try:
        await queue.consume(on_message)
        await asyncio.Future()
    finally:
        # CLEAN SHUTDOWN: stops GPU worker & waits for it
        stop_gpu_worker()
        join_gpu_worker(gpu_thread)


if __name__ == "__main__":
    asyncio.run(start_worker())
