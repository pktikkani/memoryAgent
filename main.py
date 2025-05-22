import os
from typing import List

from pydantic import BaseModel
from sqlalchemy import text
from sqlmodel import Session

from memory import MemoryManagerSQL, LongTermMemoryEntry
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import ModelMessage # For Phase 1 STM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# --- 1. Configuration ---
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
DB_PORT = os.environ.get("DB_PORT")
DB_NAME = os.environ.get("DB_NAME")

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
LTM_DATA_DIR = os.path.join(os.getcwd(), "ltm_agent_data_store_all")

OPENAI_MODEL_NAME = "gpt-4o-mini"

def get_current_user():
    username = input("Enter your username: ").strip()
    return username if username else "Anonymous"

# --- 2. Initialize MemoryManager ---
RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN = False # TRUE for 1st Phase 1, then FALSE

if RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN:
    print("RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN is True. Clearing old FAISS files...")
    # Simplified: Assumes LTM_DATA_DIR exists or will be created by MemoryManagerSQL
    # For a truly robust clear, you might delete and recreate LTM_DATA_DIR here.
    index_path = os.path.join(LTM_DATA_DIR, "my_faiss_index.idx")
    map_path = os.path.join(LTM_DATA_DIR, "faiss_id_mappings.pkl")
    if os.path.exists(index_path): os.remove(index_path)
    if os.path.exists(map_path): os.remove(map_path)
    if not os.path.exists(LTM_DATA_DIR): os.makedirs(LTM_DATA_DIR, exist_ok=True)


class MemoryContext(BaseModel):
    user_id: str
    relevant_memories: List[LongTermMemoryEntry]

agent = Agent(
    model=f"openai:{OPENAI_MODEL_NAME}",
    deps_type=MemoryContext
)


def cleanup_all_data():
    """Clean up all data from both PostgreSQL and FAISS"""
    print("⚠️  WARNING: This will delete ALL data from PostgreSQL and FAISS!")
    confirm = input("Type 'DELETE ALL' to confirm: ").strip()

    if confirm != 'DELETE ALL':
        print("Cleanup cancelled.")
        return

    try:
        # 1. Clean PostgreSQL
        with Session(memory_manager.engine) as session:
            # Delete all records from the table
            session.exec(text("DELETE FROM long_term_memories_sqlmodel"))
            session.commit()
            print("✅ PostgreSQL table cleared")

        # 2. Clean FAISS index
        memory_manager._initialize_fresh_faiss()  # Reset to empty FAISS

        # 3. Remove FAISS files from disk
        index_path = os.path.join(memory_manager.data_dir, "my_faiss_index.idx")
        map_path = os.path.join(memory_manager.data_dir, "faiss_id_mappings.pkl")

        for file_path in [index_path, map_path]:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"✅ Removed {os.path.basename(file_path)}")

        print("✅ All data cleaned successfully!")

    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

@agent.system_prompt
def base_system_prompt() -> str:
    return "You are a helpful AI assistant."


@agent.system_prompt
def dynamic_system_prompt(ctx: RunContext[MemoryContext]) -> str:
    memories = ctx.deps.relevant_memories
    if not memories:
        return ""  # No additional context if no memories

    memory_context = "\n".join([f"- {mem.data.get('text', '')}" for mem in memories])
    return f"Relevant user history:\n{memory_context}\n\nRespond helpfully using this context."


memory_manager = MemoryManagerSQL(
    db_url=DATABASE_URL,
    embedding_model_name=EMBEDDING_MODEL_NAME,
    data_dir=LTM_DATA_DIR,
)
print(f"MemoryManagerSQL setup for DB: {DB_NAME}")


# --- Phase 1: Store Every User Utterance ---

def run_phase1_store_everything_interactive():
    print("\n--- PHASE 1: CHAT & STORE ALL YOUR INPUTS to LTM ---")
    print("Everything you type will be stored. Type 'quit' to end.")
    current_user = get_current_user()

    current_stm: List[ModelMessage] = []

    while True:
        user_input = input(f"{current_user} (You): ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip(): # Skip empty inputs
            continue

        empty_context = MemoryContext(user_id=current_user, relevant_memories=[])

        response_obj = agent.run_sync(user_prompt=user_input, message_history=current_stm, deps=empty_context)
        assistant_reply = response_obj.output
        print(f"Agent: {assistant_reply}")
        current_stm = response_obj.all_messages()

        # Store the user's raw input as an LTM entry
        ltm_entry = LongTermMemoryEntry(
            user_id=current_user,
            content_type="user_utterance",
            data={"text": user_input}, # Store the raw text
            embedding_source_text=f"User {current_user} previously said: {user_input}", # Contextualize for embedding
            tags=["utterance", "phase1_chat"]
        )
        try:
            memory_manager.add_memory(ltm_entry)
            print(f"    [LTM Stored: '{user_input[:50]}...']")
        except Exception as e:
            print(f"    [ERROR storing to LTM: {e}]")
    print("\n--- PHASE 1: LTM Population Complete ---")


def run_phase2_dynamic_prompt():
    # agent = Agent(model=f"openai:{OPENAI_MODEL_NAME}", deps_type=MemoryContext)

    current_user = get_current_user()

    while True:
        user_query = input(f"{current_user} (You): ")
        if user_query.lower() == 'quit': break

        # Retrieve relevant memories
        memories = memory_manager.retrieve_memories_semantic(user_query, current_user, k=10)

        # Run with dynamic context - much simpler!
        result = agent.run_sync(
            user_prompt=user_query,
            deps=MemoryContext(user_id=current_user, relevant_memories=memories)
        )

        print(f"Agent: {result.output}")



if __name__ == "__main__":
    print("Ultra-Simple Interactive LTM Test (Storing All User Inputs)")
    if RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN:
        print("IMPORTANT: RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN is True.")

    while True:
        choice = input("\nChoose mode: [1] Chat & Store LTM (Phase 1), [2] Chat & Recall LTM (Phase 2), [3] Cleanup All Data, [q] Quit: ").strip().lower()
        if choice == '1':
            pass
            run_phase1_store_everything_interactive()
        elif choice == '2':
            if RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN:
                print("Warning: RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN is True. Phase 2 might not find LTM if Phase 1 wasn't just run.")
            run_phase2_dynamic_prompt()
        elif choice == '3':
            cleanup_all_data()
        elif choice == 'q':
            break
        else:
            print("Invalid choice.")

    if hasattr(memory_manager, 'close') and callable(memory_manager.close):
        memory_manager.close()
        print("MemoryManager closed.")
    print("Exiting script.")