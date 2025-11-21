from pymongo import MongoClient

from utils.config import DB_NAME, MONGO_URI, TENDERS_COLLECTION, VECTOR_COLLECTION

mongo = MongoClient(MONGO_URI)
db = mongo[DB_NAME]

vector_collection = db[VECTOR_COLLECTION]
tenders_collection = db[TENDERS_COLLECTION]


def store_embeddings_in_db(embeddings, document_name, tender_id):
    try:
        vector_collection.insert_many(embeddings)
    except Exception as e:
        print(f"❌ Mongo Insert Error: {e}")


def get_tender_ids(min_value):
    cursor = tenders_collection.find({"tender_value": {"$gte": min_value}}, {"_id": 1})
    return [str(doc["_id"]) for doc in cursor]
