import json
import os

import pika
import psycopg2

from processor import process_single_tender
from utils.config import RABBIT_URL

POSTGRES_DSN = os.getenv("POSTGRES_CONN_STRING")


def connect_db():
    return psycopg2.connect(POSTGRES_DSN)


def claim_job(conn, job_id):
    """
    Atomically claim a job.
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


async def on_message(ch, method, properties, body):
    job_msg = json.loads(body)
    job_id = job_msg["job_id"]

    # print("RAW MESSAGE BODY:", body)
    # print("PARSED:", job_msg)
    # print("JOB ID TYPE:", type(job_id))

    conn = connect_db()

    # claim job
    claimed = claim_job(conn, job_id)
    if not claimed:
        ch.basic_ack(method.delivery_tag)
        return

    payload_json, attempts = claimed
    payload = payload_json

    try:
        # DO THE PYTHON LOGIC
        python_result = await process_single_tender(payload)
        complete_job(conn, job_id, python_result)
        ch.basic_ack(method.delivery_tag)
    except Exception as e:
        fail_job(conn, job_id, str(e))
        ch.basic_ack(method.delivery_tag)

    conn.close()


def start_worker():
    print(RABBIT_URL)
    connection = pika.BlockingConnection(pika.URLParameters(RABBIT_URL))
    channel = connection.channel()

    channel.queue_declare(queue="jobs.python", durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="jobs.python", on_message_callback=on_message)

    print("Python worker started. Waiting for messages...")
    channel.start_consuming()


if __name__ == "__main__":
    start_worker()
