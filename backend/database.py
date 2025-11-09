"""
Database connection and initialization
"""
import os
from motor.motor_asyncio import AsyncIOMotorClient


# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]
