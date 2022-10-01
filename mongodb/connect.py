import pymongo

def get_mongo_client(username, password):
    return pymongo.MongoClient(
    f"mongodb+srv://{username}:{password}@cluster0.2atxk.mongodb.net/?retryWrites=true&w=majority"
)

def get_db(client):
    return client.nameofdb