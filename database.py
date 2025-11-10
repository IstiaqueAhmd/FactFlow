import pymongo
from pymongo import MongoClient
from typing import List, Optional, Dict
from datetime import datetime
from bson import ObjectId
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

