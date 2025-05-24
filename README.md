# Long-Term Memory for Pydantic AI Agents

A production-ready implementation of persistent memory for AI agents using Pydantic AI, PostgreSQL, and FAISS. This system enables AI agents to remember conversations, learn user preferences, and provide increasingly personalized assistance over time.

## 🚀 Features

- **Persistent Memory**: Store and retrieve co2nversation history across sessions
- **Semantic Search**: Find relevant memories using vector similarity with FAISS
- **User Isolation**: Multi-user support with private memory contexts
- **Flexible Storage**: JSONB fields for various content types and use cases
- **Production Ready**: Atomic operations, error handling, and graceful cleanup
- **Type Safe**: Built with Pydantic AI for full type safety and validation

## 🏗️ Architecture

The system uses a hybrid storage approach:

- **Pydantic AI**: Modern framework for building production-ready AI agents
- **PostgreSQL**: Stores structured memory data with JSONB and array support
- **FAISS**: Handles vector embeddings for fast semantic simil2]][[[arity search
- **Sentence Transformers**: Converts text to high-quality embeddings

## 📋 Prerequisites

- Python 3.8+
- PostgreSQL database
- OpenAI API key (or other supported LLM provider)

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-long-term-memory.git
   cd ai-long-term-memory
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL database**
   
   **Option A: Using Docker (Recommended for development)**
   ```bash
   # Run PostgreSQL in a Docker container
   docker run --name ltm-postgres \
     -e POSTGRES_USER=ltm_user \
     -e POSTGRES_PASSWORD=ltm_password \
     -e POSTGRES_DB=ltm_database \
     -p 5432:5432 \
     -d postgres:15
   ```
   
   **Option B: Local PostgreSQL installation**
5. **Run the application**

4. **Set up environment variables**
4. **Set up environment variables**
   ```bash
   # If using Docker setup above
   export DB_USER="ltm_user"
   export DB_PASSWORD="ltm_password"
   export DB_HOST="localhost"
   export DB_PORT="5432"
   export DB_NAME="ltm_database"
   export OPENAI_API_KEY="your_openai_api_key"
   ```

5. **Create PostgreSQL database**
   ```sql
   CREATE DATABASE ltm_database;
   ```

## 🐳 Docker Setup (Alternative)

For a complete containerized setup, you can use Docker Compose:

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: ltm_user
      POSTGRES_PASSWORD: ltm_password
      POSTGRES_DB: ltm_database
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

```bash
# Start the database
docker-compose up -d

# Stop the database  
docker-compose down
```

## 🚦 Quick Start

1. **Run the interactive demo**
   ```bash
   python main.py
   ```

2. **Choose Phase 1: Chat & Store LTM**
   - Have conversations with the AI
   - Every input gets stored in long-term memory
   - Build up your personal knowledge base

3. **Switch to Phase 2: Chat & Recall LTM**
   - Experience contextual conversations
   - The AI remembers your preferences and history
   - Watch responses become increasingly personalized

## 💾 Core Components

### Memory Manager (`memory.py`)
The main class that orchestrates PostgreSQL and FAISS operations:

```python
memory_manager = MemoryManagerSQL(
    db_url=DATABASE_URL,
    embedding_model_name='all-MiniLM-L6-v2',
    data_dir="./memory_data"
)
```

### Memory Entry Model
```python
class LongTermMemoryEntry(BaseModel):
    memory_id: uuid.UUID
    user_id: Optional[str] = None
    timestamp: datetime.datetime
    content_type: str
    data: Dict[str, Any]
    tags: List[str]
    embedding_source_text: str
```

### Pydantic AI Integration
```python
class MemoryContext(BaseModel):
    user_id: str
    relevant_memories: List[LongTermMemoryEntry]

agent = Agent(
    model="openai:gpt-4o-mini",
    deps_type=MemoryContext
)
```

## 🔍 Usage Examples

### Basic Memory Storage
```python
# Store a user utterance
memory_entry = LongTermMemoryEntry(
    user_id="john_doe",
    content_type="user_utterance",
    data={"text": "I prefer Python for backend development"},
    embedding_source_text="User prefers Python for backend development",
    tags=["preference", "programming"]
)
memory_manager.add_memory(memory_entry)
```

### Semantic Memory Retrieval
```python
# Find relevant memories
memories = memory_manager.retrieve_memories_semantic(
    query="backend programming preferences",
    user_id="john_doe",
    k=5
)
```

### Tag-Based Retrieval
```python
# Find memories by tags
project_memories = memory_manager.retrieve_memories_keyword(
    tags_query=["project", "ml"],
    user_id="john_doe",
    k=10
)
```

## 📊 Two-Phase Approach

### Phase 1: Learning
- Store every user interaction
- Build comprehensive knowledge base
- No memory context in responses

### Phase 2: Application  
- Retrieve relevant memories for each query
- Provide contextual, personalized responses
- Demonstrate memory-based understanding

## ⚙️ Configuration

### Database Settings
```python
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
```

### Embedding Model Options
- `all-MiniLM-L6-v2`: Fast, good for general use (384 dimensions)
- `all-mpnet-base-v2`: Better quality, slower (768 dimensions)
- `sentence-transformers/all-distilroberta-v1`: Good balance

### FAISS Configuration
- Uses `IndexIDMap` with `IndexFlatL2` for exact similarity search
- Automatic batching and persistence
- Atomic file operations for data consistency

## 🏭 Production Considerations

### Performance Optimizations
- Batch FAISS operations (configurable threshold)
- PostgreSQL indexing on `user_id`
- Atomic file writes for FAISS persistence
- Connection pooling ready

### Error Handling
- Graceful degradation if memory retrieval fails
- Automatic cleanup on shutdown
- Transaction rollback on errors
- Signal handling for clean exits

### Monitoring
```python
# Add logging for production deployments
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

## 🔒 Security & Privacy

- User memory isolation by default
- Optional global memories (user_id = NULL)
- No cross-user memory leakage
- Environment-based configuration

## 🧪 Development

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest tests/
```

### Code Style
```bash
# Format code
black .
isort .

# Type checking
mypy .
```

## 📚 Documentation

For detailed implementation guides, check out our blog series:

- **[Part 1: Building Long-Term Memory for AI Agents](https://x.com/pktikkani/status/1925388817303032214)** - Architecture and Implementation
- **[Part 2: Using Long-Term Memory with Pydantic AI](https://x.com/pktikkani/status/1926178186108305891)** - Practical Usage and Integration



## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Pydantic AI](https://github.com/pydantic/pydantic-ai) for the excellent agent framework
- [FAISS](https://github.com/facebookresearch/faiss) for high-performance vector search
- [Sentence Transformers](https://github.com/UKPLab/sentence-transformers) for quality embeddings

## 📞 Support

- 📧 Email: [your-email@company.com]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/ai-long-term-memory/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/ai-long-term-memory/discussions)

---

**Built with ❤️ for the AI community**