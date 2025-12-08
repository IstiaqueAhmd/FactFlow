from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, Index, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import List, Optional, Dict
from datetime import datetime
import os
import uuid
from dotenv import load_dotenv
from models import CheckResponse

# Load environment variables
load_dotenv()

# Create declarative base
Base = declarative_base()

class FactCheck(Base):
    """SQLAlchemy model for fact checks table"""
    __tablename__ = "fact_checks"
    
    uid = Column(String(255), primary_key=True)
    user_id = Column(String(255), nullable=False, index=True)
    verdict = Column(String(50), nullable=False, index=True)
    confidence = Column(Float, nullable=False)
    claim = Column(Text, nullable=False)
    conclusion = Column(Text, nullable=False)
    evidence = Column(JSON, nullable=False)
    sources = Column(JSON, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    
    # Create composite index for user_id and timestamp
    __table_args__ = (
        Index('idx_user_timestamp', 'user_id', 'timestamp'),
    )

class Database:
    def __init__(self):
        # Get PostgreSQL connection string from environment variable
        self.connection_string = os.getenv("DATABASE_URL")
        if not self.connection_string:
            raise ValueError("DATABASE_URL environment variable is not set")
        
        # Initialize SQLAlchemy engine and session
        self.engine = None
        self.SessionLocal = None
        
    def connect(self):
        """Establish connection to PostgreSQL"""
        try:
            # Create engine
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,  # Verify connections before using them
                pool_size=5,
                max_overflow=10
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Test the connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            print("Successfully connected to PostgreSQL")
            
        except Exception as e:
            print(f"Error connecting to PostgreSQL: {e}")
            raise
    
    def disconnect(self):
        """Close PostgreSQL connection"""
        if self.engine:
            self.engine.dispose()
    
    def init_db(self):
        """Initialize the database with required tables and indexes"""
        try:
            self.connect()
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            print("Database initialized successfully with tables and indexes")
            
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get a new database session"""
        if not self.SessionLocal:
            self.connect()
        return self.SessionLocal()
    
    def save_fact_check(self, user_id: str, check_result: CheckResponse) -> str:
        """
        Save a fact check result to the database.
        
        Args:
            user_id: The user ID who performed the fact check
            check_result: The CheckResponse object containing the fact check results
            
        Returns:
            The inserted record ID as a string
        """
        session = self.get_session()
        try:
            # Generate unique ID
            fact_check_uid = str(uuid.uuid4())
            
            # Create FactCheck record
            fact_check = FactCheck(
                uid=fact_check_uid,
                user_id=user_id,
                verdict=check_result.verdict,
                confidence=check_result.confidence,
                claim=check_result.claim,
                conclusion=check_result.conclusion,
                evidence=check_result.evidence,
                sources=[{"title": s.title, "url": s.url} for s in check_result.sources],
                timestamp=check_result.timestamp
            )
            
            # Add and commit
            session.add(fact_check)
            session.commit()
            
            print(f"Fact check saved successfully with ID: {fact_check_uid}")
            
            return fact_check_uid
            
        except Exception as e:
            session.rollback()
            print(f"Error saving fact check to database: {e}")
            raise
        finally:
            session.close()

    def get_fact_checks(self, user_id: str, limit: Optional[int] = None, verdict: Optional[str] = None) -> List[Dict]:
        """
        Retrieve past fact check results for a given user.
        
        Args:
            user_id: The user ID whose results to retrieve
            limit: Maximum number of results to retrieve
            verdict: Optional filter for verdict (e.g., "true", "false"). If None, returns all fact-checks.
        """
        session = self.get_session()
        try:
            # Query fact checks for the user, ordered by timestamp descending
            query = session.query(FactCheck).filter(FactCheck.user_id == user_id)
            
            # Add verdict filter if provided
            if verdict is not None and verdict.strip():
                query = query.filter(FactCheck.verdict == verdict)
            
            query = query.order_by(FactCheck.timestamp.desc())
            
            if limit:
                query = query.limit(limit)
            
            results = []
            for fact_check in query.all():
                results.append({
                    "_id": fact_check.uid,
                    "user_id": fact_check.user_id,
                    "verdict": fact_check.verdict,
                    "confidence": fact_check.confidence,
                    "claim": fact_check.claim,
                    "conclusion": fact_check.conclusion,
                    "evidence": fact_check.evidence,
                    "sources": fact_check.sources,
                    "timestamp": fact_check.timestamp
                })
            
            return results
            
        except Exception as e:
            print(f"Error retrieving fact checks from database: {e}")
            raise
        finally:
            session.close()
    
    def delete_fact_check(self, uid: str, user_id: str) -> bool:
        """
        Delete a fact check result from the database.
        
        Args:
            uid: The unique ID of the fact check to delete
            user_id: The user ID who owns the fact check (for authorization)
            
        Returns:
            True if deleted successfully, False if not found or unauthorized
        """
        session = self.get_session()
        try:
            # Find the fact check
            fact_check = session.query(FactCheck).filter(
                FactCheck.uid == uid,
                FactCheck.user_id == user_id
            ).first()
            
            if not fact_check:
                return False
            
            # Delete the fact check
            session.delete(fact_check)
            session.commit()
            
            print(f"Fact check deleted successfully: {uid}")
            return True
            
        except Exception as e:
            session.rollback()
            print(f"Error deleting fact check from database: {e}")
            raise
        finally:
            session.close()
        
