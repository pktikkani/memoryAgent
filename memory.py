import datetime
import os
import pickle
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel as PydanticBaseModel
from sqlmodel import Field, SQLModel, Column, ARRAY, String # Added ARRAY, String
from sqlalchemy.dialects.postgresql import JSONB, UUID as PG_UUID # For PostgreSQL specific types
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sqlmodel import create_engine, Session, select # Import select
import time
import signal
import atexit

# This is your Pydantic model for API/application logic
class LongTermMemoryEntry(PydanticBaseModel):
    memory_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    user_id: Optional[str] = None
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow)
    content_type: str
    data: Dict[str, Any]
    tags: List[str] = Field(default_factory=list)
    embedding_source_text: str
    embedding: Optional[List[float]] = Field(default=None, exclude=True)

    class Config:
        from_attributes = True
        frozen = False # Keep this if you might modify instances before saving
        arbitrary_types_allowed = True


# This is your SQLModel for database persistence
class LongTermMemoryEntrySQL(SQLModel, table=True):
    __tablename__ = "long_term_memories_sqlmodel" # Define table name

    memory_id: uuid.UUID = Field(default_factory=uuid.uuid4, sa_column=Column(PG_UUID(as_uuid=True), primary_key=True, unique=True, nullable=False))
    user_id: Optional[str] = Field(default=None, index=True)
    timestamp: datetime.datetime = Field(default_factory=datetime.datetime.utcnow, sa_column_kwargs={"default": "now"})
    content_type: str
    data: Dict[str, Any] = Field(sa_column=Column(JSONB)) # Use JSONB for PostgreSQL
    tags: List[str] = Field(default_factory=list, sa_column=Column(ARRAY(String))) # PostgreSQL array of text
    embedding_source_text: str


class MemoryManagerSQL:
    def __init__(self, db_url: str, embedding_model_name: str = 'all-MiniLM-L6-v2', data_dir=None):

        try:
            self.engine = create_engine(db_url)
            self._create_db_and_tables()
            print("Successfully connected to PostgreSQL via SQLModel.")
        except Exception as e:
            print(f"Error connecting to PostgreSQL via SQLModel: {e}")
            raise

        try:
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            print(f"Embedding model '{embedding_model_name}' loaded with dimension {self.embedding_dim}.")
        except Exception as e:
            print(f"Error loading embedding model '{embedding_model_name}': {e}")
            raise
        self.data_dir = data_dir or os.path.join(os.getcwd(), "memory_data")

        # Create directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)

        index_path = os.path.join(self.data_dir, "my_faiss_index.idx")
        map_path = os.path.join(self.data_dir, "faiss_id_mappings.pkl")

        if os.path.exists(index_path) and os.path.exists(map_path):
            try:
                self.faiss_index = faiss.read_index(index_path)
                with open(map_path, "rb") as f:
                    id_mappings = pickle.load(f)
                self._faiss_id_to_memory_uuid = id_mappings.get("faiss_id_to_memory_uuid", {})
                self._memory_uuid_to_faiss_id = id_mappings.get("memory_uuid_to_faiss_id", {})
                self._next_available_faiss_id = id_mappings.get("next_available_faiss_id", 0)
                print(
                    f"FAISS index and mappings loaded from {self.data_dir}. Next FAISS ID: {self._next_available_faiss_id}")
            except Exception as e:
                print(f"Error loading FAISS data, initializing fresh: {e}")
                self._initialize_fresh_faiss()  # A method to set up new FAISS structures
        else:
            print("No existing FAISS data found, initializing fresh.")
            self._initialize_fresh_faiss()

        print("MemoryManagerSQL initialized.")
        self._unsaved_changes = 0
        self._batch_save_threshold = 5  # Save every 5 additions
        self._last_save_time = time.time()
        self._save_interval_seconds = 300  # Save every 5 minutes regardless
        atexit.register(self.close)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"Received signal {signum}, cleaning up...")
        self.close()
        exit(0)

    def _initialize_fresh_faiss(self):
        self.faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(self.embedding_dim))
        self._memory_uuid_to_faiss_id: Dict[uuid.UUID, int] = {}
        self._faiss_id_to_memory_uuid: Dict[int, uuid.UUID] = {}
        self._next_available_faiss_id: int = 0

    def _create_db_and_tables(self):
        try:
            SQLModel.metadata.create_all(self.engine)
            print("SQLModel table 'long_term_memories_sqlmodel' initialized/verified.")
        except Exception as e:
            print(f"Error initializing SQLModel database table: {e}")
            raise

    def _save_faiss_to_disk(self):
        """Save FAISS index and mappings to disk"""
        try:
            # Use temporary files for atomic writes
            index_path = os.path.join(self.data_dir, "my_faiss_index.idx")
            temp_index_path = index_path + ".tmp"

            map_path = os.path.join(self.data_dir, "faiss_id_mappings.pkl")
            temp_map_path = map_path + ".tmp"

            # Write to temporary files first
            faiss.write_index(self.faiss_index, temp_index_path)

            mappings_and_next_id_to_save = {
                "faiss_id_to_memory_uuid": self._faiss_id_to_memory_uuid,
                "memory_uuid_to_faiss_id": self._memory_uuid_to_faiss_id,
                "next_available_faiss_id": self._next_available_faiss_id
            }

            with open(temp_map_path, "wb") as f:
                pickle.dump(mappings_and_next_id_to_save, f)

            # Atomic rename (on most filesystems)
            os.rename(temp_index_path, index_path)
            os.rename(temp_map_path, map_path)

            print(f"FAISS data saved to disk (Index: {self.faiss_index.ntotal} vectors)")

        except Exception as e:
            print(f"Error saving FAISS data to disk: {e}")
            # Clean up temp files if they exist
            for temp_path in [temp_index_path, temp_map_path]:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except:
                        raise

    def add_memory(self, memory_item_pydantic: LongTermMemoryEntry):
        # Convert Pydantic model to SQLModel instance
        memory_item_sql = LongTermMemoryEntrySQL.model_validate(memory_item_pydantic)

        with Session(self.engine) as session:
            try:
                # Upsert logic for SQLModel
                existing = session.get(LongTermMemoryEntrySQL, memory_item_sql.memory_id)
                if existing:
                    for key, value in memory_item_pydantic.model_dump(exclude_unset=True).items():
                        if key != "embedding":  # Don't try to set embedding on SQL model
                            setattr(existing, key, value)
                    session.add(existing)
                    print(f"Updated memory {memory_item_sql.memory_id} in PostgreSQL.")
                else:
                    session.add(memory_item_sql)
                    print(f"Stored new memory {memory_item_sql.memory_id} in PostgreSQL.")
                session.commit()
                session.refresh(memory_item_sql)  # To get any DB defaults like timestamp
            except Exception as e:
                session.rollback()
                print(f"Error storing memory {memory_item_sql.memory_id} in PostgreSQL: {e}")
                return

        # Add to FAISS
        if memory_item_pydantic.embedding_source_text:
            try:
                embedding = self.embedding_model.encode(memory_item_pydantic.embedding_source_text)
                embedding_np = np.array([embedding], dtype='float32')

                # Manage FAISS IDs
                faiss_id_to_use = self._memory_uuid_to_faiss_id.get(memory_item_pydantic.memory_id)
                # Track unsaved changes
                if faiss_id_to_use is None:  # New entry for FAISS
                    faiss_id_to_use = self._next_available_faiss_id
                    self._memory_uuid_to_faiss_id[memory_item_pydantic.memory_id] = faiss_id_to_use
                    self._faiss_id_to_memory_uuid[faiss_id_to_use] = memory_item_pydantic.memory_id
                    self._next_available_faiss_id += 1
                    self.faiss_index.add_with_ids(embedding_np, np.array([faiss_id_to_use], dtype='int64'))
                    print(
                        f"Added embedding for memory {memory_item_pydantic.memory_id} to FAISS (FAISS ID: {faiss_id_to_use}).")
                else:  # Update existing vector (requires remove then add, or an updatable index type)
                    self.faiss_index.add_with_ids(embedding_np, np.array([faiss_id_to_use], dtype='int64'))
                    print(
                        f"Updated/Re-added embedding for memory {memory_item_pydantic.memory_id} in FAISS (FAISS ID: {faiss_id_to_use}).")

                # NOW track unsaved changes and check if we should save (AFTER adding to FAISS)

                self._unsaved_changes += 1

                # Check if we should save
                should_save = (
                        self._unsaved_changes >= self._batch_save_threshold or
                        time.time() - self._last_save_time > self._save_interval_seconds
                )

                if should_save:
                    self._save_faiss_to_disk()
                    self._unsaved_changes = 0
                    self._last_save_time = time.time()

            except Exception as e:
                print(f"Error adding/updating embedding for memory {memory_item_pydantic.memory_id} to FAISS: {e}")


    def retrieve_memories_semantic(self, query: str, user_id: Optional[str] = None, k: int = 3) -> List[
        LongTermMemoryEntry]:
        if not query or self.faiss_index.ntotal == 0:
            return []

        print(f"Performing semantic search for query: '{query[:30]}...' for user '{user_id}'")
        query_embedding = self.embedding_model.encode(query)
        query_embedding_np = np.array([query_embedding], dtype='float32')

        # Search FAISS
        # k_faiss = min(k * 2, self.faiss_index.ntotal) # Retrieve more from FAISS to filter by user_id later
        k_faiss = min(k, self.faiss_index.ntotal)
        distances, faiss_ids_retrieved = self.faiss_index.search(query_embedding_np, k=k_faiss)

        retrieved_memory_uuids = []
        for faiss_id_int in faiss_ids_retrieved[0]:
            if faiss_id_int != -1:  # FAISS uses -1 for no result or if k > ntotal for some index types
                memory_uuid = self._faiss_id_to_memory_uuid.get(faiss_id_int)
                if memory_uuid:
                    retrieved_memory_uuids.append(memory_uuid)

        if not retrieved_memory_uuids:
            print("No matching UUIDs found from FAISS results.")
            return []

        # Fetch full items from PostgreSQL using SQLModel
        with Session(self.engine) as session:
            try:
                statement = select(LongTermMemoryEntrySQL).where(
                    LongTermMemoryEntrySQL.memory_id.in_(retrieved_memory_uuids))
                if user_id:
                    # Filter by user_id, allowing global memories (user_id IS NULL)
                    statement = statement.where(
                        (LongTermMemoryEntrySQL.user_id == user_id) | (LongTermMemoryEntrySQL.user_id.is_(None))
                    )

                results_sql = session.exec(statement).all()

                # Convert SQLModel instances to Pydantic instances
                retrieved_pydantic_entries: List[LongTermMemoryEntry] = []
                if results_sql:
                    for rsql_item in results_sql:
                        try:
                            # Convert SQLModel instance to a dictionary
                            item_dict = rsql_item.model_dump()  # For SQLModel based on Pydantic v2
                            # item_dict = rsql_item.dict() # If your SQLModel version uses .dict()

                            # Now validate from the dictionary
                            pydantic_entry = LongTermMemoryEntry.model_validate(item_dict)
                            retrieved_pydantic_entries.append(pydantic_entry)
                        except Exception as e:
                            print(f"Error converting SQLModel item to Pydantic LongTermMemoryEntry: {e}")
                            print(f"Problematic SQLModel item: {rsql_item}")  # Log the problematic item
                            # Optionally, try to create a partial entry or skip
                            # For debugging, you might want to raise the error to see the full traceback

                print(
                    f"Retrieved and converted {len(retrieved_pydantic_entries)} memories from PostgreSQL after FAISS.")
                return retrieved_pydantic_entries[:k]
            except Exception as e:
                print(f"Error fetching memories from PostgreSQL (SQLModel) after FAISS search: {e}")
                return []

    def retrieve_memories_keyword(self, tags_query: List[str], user_id: Optional[str] = None, k: int = 3) -> List[
        LongTermMemoryEntry]:
        print(f"Performing keyword (tag) search: {tags_query} for user '{user_id}'")
        if not tags_query:
            return []
        with Session(self.engine) as session:
            try:
                statement = select(LongTermMemoryEntrySQL)

                for tag in tags_query:
                    statement = statement.where(
                        LongTermMemoryEntrySQL.tags.any(tag))  # Requires ARRAY column type in SQLModel for .any()

                if user_id:
                    statement = statement.where(
                        (LongTermMemoryEntrySQL.user_id == user_id) | (LongTermMemoryEntrySQL.user_id == None)
                    )

                statement = statement.order_by(LongTermMemoryEntrySQL.timestamp.desc()).limit(k)
                results_sql = session.exec(statement).all()
                retrieved_pydantic_entries = [LongTermMemoryEntry.model_validate(rsql) for rsql in results_sql]
                print(f"Retrieved {len(retrieved_pydantic_entries)} memories by keyword.")
                return retrieved_pydantic_entries
            except Exception as e:
                print(f"Error fetching memories by keyword (SQLModel): {e}")
                return []

    def close(self):
        if self._unsaved_changes > 0:
            print(f"Saving {self._unsaved_changes} unsaved changes before closing...")
            self._save_faiss_to_disk()

        try:
            if hasattr(self, 'session') and self.session and hasattr(self.session, 'is_closed') and not self.session.is_closed:
                self.session.close()
                print("SQLModel session explicitly closed (if it was a long-lived instance var).")

            print(f"Memory manager closed successfully. Data saved to {self.data_dir}")
        except Exception as e:
            print(f"Error closing memory manager: {e}")

