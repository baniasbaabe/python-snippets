import pymongo

def get_collection(db):
    return db.nameofcollection

def insert_doc(collection, data):
    collection.insert_one(data)