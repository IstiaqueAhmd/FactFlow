import pymongo
from pymongo import MongoClient
from typing import List, Optional, Dict
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv
from models import CheckResponse

# Load environment variables
load_dotenv()

class Database:
    def __init__(self):
        # Get MongoDB connection string from environment variable
        self.connection_string = os.getenv("MONGODB_URI")
        self.database_name = os.getenv("DATABASE_NAME", "factflow_db")
        self.collection_name = "fact_checks"
        
        # Initialize MongoDB client
        self.client = None
        self.db = None
        self.collection = None
        
    def connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client[self.database_name]
            self.collection = self.db[self.collection_name]
            
            # Test the connection
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB")
            
        except Exception as e:
            print(f"Error connecting to MongoDB: {e}")
            raise
    
    def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
    
    def init_db(self):
        """Initialize the database with required indexes"""
        try:
            self.connect()
            
            # Create indexes for better performance
            self.collection.create_index([("user_id", 1)])
            self.collection.create_index([("timestamp", -1)])
            self.collection.create_index([("verdict", 1)])
            
            print("Database initialized successfully with indexes")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def save_fact_check(self, user_id: str, check_result: CheckResponse) -> str:
        """
        Save a fact check result to the database.
        
        Args:
            user_id: The user ID who performed the fact check
            check_result: The CheckResponse object containing the fact check results
            
        Returns:
            The inserted document ID as a string
        """
        try:
            # Convert CheckResponse to dictionary and add user_id
            document = {
                "user_id": user_id,
                "verdict": check_result.verdict,
                "confidence": check_result.confidence,
                "claim": check_result.claim,
                "conclusion": check_result.conclusion,
                "evidence": check_result.evidence,
                "sources": [{"title": s.title, "url": s.url} for s in check_result.sources],
                "timestamp": check_result.timestamp,
            }
            
            # Insert the document
            result = self.collection.insert_one(document)
            print(f"Fact check saved successfully with ID: {result.inserted_id}")
            
            return str(result.inserted_id)
            
        except Exception as e:
            print(f"Error saving fact check to database: {e}")
            raise
    