#!/usr/bin/env python3
"""
Memory Manager Module - Gives Phoenix the ability to remember and learn.
This module handles conversation history, knowledge storage, and learning from interactions.
"""

import os
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import hashlib

# These will be installed when needed
try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class MemoryManager:
    """
    Handles all memory operations for Phoenix.
    This is like Phoenix's 'brain storage' - it remembers everything important.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Memory Manager.

        Args:
            config: Configuration dictionary from phoenix_config.yaml
        """
        self.config = config
        self.logger = logging.getLogger(f"PHOENIX.Memory")

        # Paths for storage
        self.data_dir = Path(config.get('persist_directory', './data'))
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Conversation history
        self.conversation_file = self.data_dir / 'conversations.json'
        self.current_conversation = []
        self.conversation_history = self._load_conversations()

        # User profile
        self.profile_file = self.data_dir / 'user_profile.json'
        self.user_profile = self._load_profile()

        # Knowledge base (simple for now)
        self.knowledge_file = self.data_dir / 'knowledge.json'
        self.knowledge_base = self._load_knowledge()

        # Initialize vector database if available
        self.vector_db = None
        self.collection = None
        if HAS_CHROMA:
            self._init_vector_db()

        # Initialize knowledge graph if available
        self.knowledge_graph = None
        if HAS_NETWORKX:
            self._init_knowledge_graph()

        self.logger.info("Memory Manager initialized")

    # ============= CORE MEMORY FUNCTIONS =============

    def remember_conversation(self, user_input: str, ai_response: str, metadata: Dict = None):
        """
        Store a conversation turn in memory.

        Args:
            user_input: What the user said
            ai_response: What Phoenix responded
            metadata: Additional context about the conversation
        """
        entry = {
            'timestamp': datetime.now().isoformat(),
            'user': user_input,
            'assistant': ai_response,
            'metadata': metadata or {}
        }

        # Add to current conversation
        self.current_conversation.append(entry)

        # Store in vector database if available
        if self.collection:
            try:
                # Create a combined text for embedding
                combined_text = f"User: {user_input}\nAssistant: {ai_response}"

                # Generate unique ID
                entry_id = hashlib.md5(f"{entry['timestamp']}{user_input}".encode()).hexdigest()

                # Store in ChromaDB
                self.collection.add(
                    documents=[combined_text],
                    metadatas=[{
                        'timestamp': entry['timestamp'],
                        'type': 'conversation',
                        'user_input': user_input[:500],  # Truncate for metadata
                    }],
                    ids=[entry_id]
                )
                self.logger.debug(f"Stored conversation in vector DB: {entry_id}")
            except Exception as e:
                self.logger.error(f"Failed to store in vector DB: {e}")

        # Learn from the conversation
        self._learn_from_interaction(user_input, ai_response)

        self.logger.info("Remembered conversation turn")

    def get_recent_memories(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent conversation memories.

        Args:
            limit: Number of recent memories to retrieve

        Returns:
            List of recent conversation entries
        """
        if not self.conversation_history and not self.current_conversation:
            return []

        # Sort by timestamp and get most recent
        all_convos = self.conversation_history + self.current_conversation
        sorted_convos = sorted(
            all_convos,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )

        # Format for LLM consumption
        formatted = []
        for convo in sorted_convos[:limit]:
            formatted.append({
                'user': convo.get('user', ''),
                'assistant': convo.get('assistant', ''),
                'timestamp': convo.get('timestamp', '')
            })

        # Return in chronological order (oldest first)
        return list(reversed(formatted))

    def recall_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Recall similar conversations or knowledge from memory.

        Args:
            query: What to search for
            limit: Maximum number of results

        Returns:
            List of similar memories
        """
        results = []

        # Search vector database if available
        if self.collection:
            try:
                search_results = self.collection.query(
                    query_texts=[query],
                    n_results=limit
                )

                for i, doc in enumerate(search_results['documents'][0]):
                    results.append({
                        'content': doc,
                        'metadata': search_results['metadatas'][0][i],
                        'distance': search_results['distances'][0][i] if 'distances' in search_results else None
                    })

                self.logger.info(f"Found {len(results)} similar memories")
            except Exception as e:
                self.logger.error(f"Vector search failed: {e}")

        # Fallback to simple keyword search in recent conversations
        if not results:
            query_lower = query.lower()
            for conv in self.current_conversation[-20:]:  # Last 20 turns
                if query_lower in conv['user'].lower() or query_lower in conv['assistant'].lower():
                    results.append({
                        'content': f"User: {conv['user']}\nAssistant: {conv['assistant']}",
                        'metadata': {'timestamp': conv['timestamp']},
                        'distance': None
                    })

        return results[:limit]

    def learn_fact(self, fact: str, category: str = 'general', confidence: float = 1.0):
        """
        Learn a new fact and store it in the knowledge base.

        Args:
            fact: The fact to learn
            category: Category of the fact
            confidence: How confident we are in this fact (0-1)
        """
        fact_entry = {
            'fact': fact,
            'category': category,
            'confidence': confidence,
            'learned_at': datetime.now().isoformat(),
            'source': 'conversation'
        }

        # Add to knowledge base
        if category not in self.knowledge_base:
            self.knowledge_base[category] = []

        self.knowledge_base[category].append(fact_entry)

        # Store in vector DB for semantic search
        if self.collection:
            try:
                fact_id = hashlib.md5(f"{fact}{datetime.now().isoformat()}".encode()).hexdigest()
                self.collection.add(
                    documents=[fact],
                    metadatas=[{
                        'type': 'knowledge',
                        'category': category,
                        'confidence': confidence,
                        'timestamp': fact_entry['learned_at']
                    }],
                    ids=[fact_id]
                )
            except Exception as e:
                self.logger.error(f"Failed to store fact in vector DB: {e}")

        # Add to knowledge graph if available
        if self.knowledge_graph:
            self._add_to_graph(fact, category)

        self.logger.info(f"Learned new fact in category '{category}': {fact[:50]}...")
        self._save_knowledge()

    def update_user_profile(self, key: str, value: Any):
        """
        Update information about the user.

        Args:
            key: Profile attribute (e.g., 'name', 'preferences')
            value: Value to store
        """
        self.user_profile[key] = value
        self.user_profile['last_updated'] = datetime.now().isoformat()

        self._save_profile()
        self.logger.info(f"Updated user profile: {key}")

    def get_user_info(self, key: str = None) -> Any:
        """
        Get information about the user.

        Args:
            key: Specific attribute to get (None for all)

        Returns:
            User information
        """
        if key:
            return self.user_profile.get(key)
        return self.user_profile

    def get_conversation_summary(self, last_n: int = 10) -> str:
        """
        Get a summary of recent conversations.

        Args:
            last_n: Number of recent turns to summarize

        Returns:
            Summary text
        """
        if not self.current_conversation:
            return "No conversation history yet."

        recent = self.current_conversation[-last_n:]
        summary = f"Last {len(recent)} conversation turns:\n"

        for conv in recent:
            time = datetime.fromisoformat(conv['timestamp']).strftime('%H:%M:%S')
            summary += f"\n[{time}] User: {conv['user'][:100]}...\n"
            summary += f"Phoenix: {conv['assistant'][:100]}...\n"

        return summary

    # ============= PRIVATE HELPER FUNCTIONS =============

    def _init_vector_db(self):
        """Initialize ChromaDB for vector storage."""
        try:
            # Create ChromaDB client
            self.vector_db = chromadb.PersistentClient(
                path=str(self.data_dir / 'chroma'),
                settings=Settings(anonymized_telemetry=False)
            )

            # Create or get collection
            collection_name = self.config.get('collection_name', 'phoenix_memory')
            self.collection = self.vector_db.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Phoenix AI memory storage"}
            )

            self.logger.info("Vector database initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize vector DB: {e}")
            self.logger.info("Will work without vector search capabilities")

    def _init_knowledge_graph(self):
        """Initialize NetworkX knowledge graph."""
        try:
            graph_file = self.data_dir / 'knowledge_graph.pkl'

            if graph_file.exists():
                with open(graph_file, 'rb') as f:
                    self.knowledge_graph = pickle.load(f)
                self.logger.info(f"Loaded knowledge graph with {len(self.knowledge_graph.nodes)} nodes")
            else:
                self.knowledge_graph = nx.DiGraph()
                self.logger.info("Created new knowledge graph")

        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge graph: {e}")

    def _add_to_graph(self, fact: str, category: str):
        """Add a fact to the knowledge graph."""
        if not self.knowledge_graph:
            return

        try:
            # Add category node if it doesn't exist
            if category not in self.knowledge_graph:
                self.knowledge_graph.add_node(category, type='category')

            # Add fact node
            fact_id = f"fact_{len(self.knowledge_graph.nodes)}"
            self.knowledge_graph.add_node(
                fact_id,
                type='fact',
                content=fact,
                timestamp=datetime.now().isoformat()
            )

            # Connect fact to category
            self.knowledge_graph.add_edge(category, fact_id)

            # Save graph
            graph_file = self.data_dir / 'knowledge_graph.pkl'
            with open(graph_file, 'wb') as f:
                pickle.dump(self.knowledge_graph, f)

        except Exception as e:
            self.logger.error(f"Failed to add to knowledge graph: {e}")

    def _learn_from_interaction(self, user_input: str, ai_response: str):
        """
        Analyze interaction and learn patterns.
        This is where Phoenix gets smarter over time.
        """
        try:
            # Learn about user preferences
            user_lower = user_input.lower()

            # Detect commands the user likes
            if any(word in user_lower for word in ['list', 'show', 'display']):
                self._increment_preference('prefers_listings', 1)

            # Detect topics of interest
            if any(word in user_lower for word in ['python', 'code', 'program']):
                self._increment_preference('interested_in_programming', 1)

            # Detect communication style
            if len(user_input) < 20:
                self._increment_preference('prefers_brief_commands', 1)

            # Store successful commands
            if 'error' not in ai_response.lower():
                self._add_successful_pattern(user_input)

        except Exception as e:
            self.logger.error(f"Learning failed: {e}")

    def _increment_preference(self, pref_key: str, amount: int = 1):
        """Increment a user preference counter."""
        if 'preferences' not in self.user_profile:
            self.user_profile['preferences'] = {}

        current = self.user_profile['preferences'].get(pref_key, 0)
        self.user_profile['preferences'][pref_key] = current + amount

    def _add_successful_pattern(self, pattern: str):
        """Remember successful command patterns."""
        if 'successful_patterns' not in self.user_profile:
            self.user_profile['successful_patterns'] = []

        # Don't duplicate
        if pattern not in self.user_profile['successful_patterns']:
            self.user_profile['successful_patterns'].append(pattern)
            # Keep only last 100
            self.user_profile['successful_patterns'] = self.user_profile['successful_patterns'][-100:]

    # ============= PERSISTENCE FUNCTIONS =============

    def _load_conversations(self) -> List[Dict]:
        """Load conversation history from disk."""
        if self.conversation_file.exists():
            try:
                with open(self.conversation_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load conversations: {e}")
        return []

    def _load_profile(self) -> Dict:
        """Load user profile from disk."""
        if self.profile_file.exists():
            try:
                with open(self.profile_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load profile: {e}")

        # Return default profile
        return {
            'created_at': datetime.now().isoformat(),
            'preferences': {},
            'facts_about_user': []
        }

    def _load_knowledge(self) -> Dict:
        """Load knowledge base from disk."""
        if self.knowledge_file.exists():
            try:
                with open(self.knowledge_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load knowledge: {e}")
        return {}

    def save(self):
        """Save all memory to disk."""
        try:
            # Save current conversation
            all_conversations = self.conversation_history + self.current_conversation
            with open(self.conversation_file, 'w') as f:
                json.dump(all_conversations, f, indent=2)

            # Save profile
            self._save_profile()

            # Save knowledge
            self._save_knowledge()

            self.logger.info("Memory saved to disk")
        except Exception as e:
            self.logger.error(f"Failed to save memory: {e}")

    def _save_profile(self):
        """Save user profile."""
        with open(self.profile_file, 'w') as f:
            json.dump(self.user_profile, f, indent=2)

    def _save_knowledge(self):
        """Save knowledge base."""
        with open(self.knowledge_file, 'w') as f:
            json.dump(self.knowledge_base, f, indent=2)

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        stats = {
            'total_conversations': len(self.conversation_history) + len(self.current_conversation),
            'current_session_turns': len(self.current_conversation),
            'knowledge_categories': len(self.knowledge_base),
            'total_facts': sum(len(facts) for facts in self.knowledge_base.values()),
            'user_preferences': len(self.user_profile.get('preferences', {})),
        }

        if self.collection:
            try:
                stats['vector_memories'] = self.collection.count()
            except:
                stats['vector_memories'] = 0

        if self.knowledge_graph:
            stats['knowledge_graph_nodes'] = len(self.knowledge_graph.nodes)
            stats['knowledge_graph_edges'] = len(self.knowledge_graph.edges)

        return stats