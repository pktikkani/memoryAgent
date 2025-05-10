import os
from typing import List
from memory import MemoryManagerSQL, LongTermMemoryEntry
from pydantic_ai import Agent
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
USER_ID = "User-1"

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

    agent_phase1 = Agent(
        model=f"openai:{OPENAI_MODEL_NAME}",
        system_prompt="You are a friendly chat partner."
    )
    current_stm: List[ModelMessage] = []

    while True:
        user_input = input(f"{USER_ID} (You): ")
        if user_input.lower() == 'quit':
            break
        if not user_input.strip(): # Skip empty inputs
            continue

        response_obj = agent_phase1.run_sync(user_prompt=user_input, message_history=current_stm)
        assistant_reply = response_obj.output
        print(f"Agent: {assistant_reply}")
        current_stm = response_obj.all_messages()

        # Store the user's raw input as an LTM entry
        ltm_entry = LongTermMemoryEntry(
            user_id=USER_ID,
            content_type="user_utterance",
            data={"text": user_input}, # Store the raw text
            embedding_source_text=f"User {USER_ID} previously said: {user_input}", # Contextualize for embedding
            tags=["utterance", "phase1_chat"]
        )
        try:
            memory_manager.add_memory(ltm_entry)
            print(f"    [LTM Stored: '{user_input[:50]}...']")
        except Exception as e:
            print(f"    [ERROR storing to LTM: {e}]")
    print("\n--- PHASE 1: LTM Population Complete ---")

# --- Phase 2: Recall LTM ---
def run_phase2_recall_everything_interactive():
    print("\n--- PHASE 2: CHAT & RECALL LTM ---")
    print("Ask questions. The agent will be given your past relevant statements as context. Type 'quit' to end.")

    while True:
        user_query = input(f"{USER_ID} (You): ")
        if user_query.lower() == 'quit':
            break
        if not user_query.strip():
            continue

        print("    Retrieving relevant past statements from LTM...")
        retrieved_ltm_items = memory_manager.retrieve_memories_semantic(
            query=user_query, user_id=USER_ID, k=3 # Retrieve top 3 past user utterances
        )

        if retrieved_ltm_items:
            ltm_context_for_prompt = "\n\n=== Some of your relevant past statements were: ===\n"
            for item in retrieved_ltm_items:
                # Assuming item.data is {"text": "original user input"}
                past_utterance = item.data.get("text", "[Could not extract past statement text]")
                ltm_context_for_prompt += f"- \"{past_utterance}\"\n"
            ltm_context_for_prompt += "====================================================\n"
            print(f"    [LTM Context for Agent (Snippet): {ltm_context_for_prompt[:200].strip()}...]")
        else:
            print("    [No specific past statements found in LTM for this question.]")
            ltm_context_for_prompt = "\n[No specific relevant background information from your past statements was found.]\n"

        recall_agent_system_prompt = (
            "You are a helpful assistant. The user is chatting with you again after a break. "
            "To help you understand the context, here are some relevant things the user said in previous conversations:"
            f"{ltm_context_for_prompt}"
            "Now, please answer the user's current question based on this context if it's helpful, "
            "otherwise use your general knowledge."
        )

        recall_agent = Agent(
            model=f"openai:{OPENAI_MODEL_NAME}",
            system_prompt=recall_agent_system_prompt
        )

        print("    Agent (with LTM context) processing...")
        response_obj = recall_agent.run_sync(user_prompt=user_query)
        assistant_reply = response_obj.output
        print(f"Agent: {assistant_reply}")

    print("\n--- PHASE 2: LTM Recall Test Complete ---")

if __name__ == "__main__":
    print("Ultra-Simple Interactive LTM Test (Storing All User Inputs)")
    if RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN:
        print("IMPORTANT: RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN is True.")

    while True:
        choice = input("\nChoose mode: [1] Chat & Store LTM (Phase 1), [2] Chat & Recall LTM (Phase 2), [q] Quit: ").strip().lower()
        if choice == '1':
            run_phase1_store_everything_interactive()
        elif choice == '2':
            if RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN:
                print("Warning: RECREATE_DB_AND_FAISS_ON_FIRST_PHASE1_RUN is True. Phase 2 might not find LTM if Phase 1 wasn't just run.")
            run_phase2_recall_everything_interactive()
        elif choice == 'q':
            break
        else:
            print("Invalid choice.")

    if hasattr(memory_manager, 'close') and callable(memory_manager.close):
        memory_manager.close()
        print("MemoryManager closed.")
    print("Exiting script.")