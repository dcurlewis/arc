"""
Enhanced Embedding System for ARC
Generates sophisticated embeddings that incorporate entity and relationship information
for improved semantic search and retrieval.
"""

import logging
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import chromadb

logger = logging.getLogger(__name__)


class EnhancedEmbeddingGenerator:
    """
    Enhanced embedding generator that creates multiple types of embeddings:
    1. Document embeddings (enhanced with entity context)
    2. Entity embeddings (entity-specific vectors)
    3. Relationship embeddings (relationship-aware vectors)
    4. Hybrid embeddings (multi-modal combinations)
    """
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.embedding_model = db_manager.embeddings
        self.chromadb_client = db_manager.chromadb
        
        # Create specialized collections using config values
        documents_collection = db_manager.config.get('chromadb.collection_documents', 'documents')
        self.collections = {
            'documents': self._get_or_create_collection(documents_collection),
            'entities': self._get_or_create_collection(f"{documents_collection}_entities"),
            'relationships': self._get_or_create_collection(f"{documents_collection}_relationships"),
            'hybrid': self._get_or_create_collection(f"{documents_collection}_hybrid")
        }
        
        logger.info("Initialized Enhanced Embedding Generator")
    
    def _get_or_create_collection(self, name: str) -> chromadb.Collection:
        """Get or create a ChromaDB collection."""
        try:
            return self.chromadb_client.get_collection(name)
        except Exception:
            # Use create_collection instead of get_or_create to avoid dynamic naming
            try:
                return self.chromadb_client.create_collection(name)
            except Exception:
                # If creation fails, try to get existing one
                return self.chromadb_client.get_collection(name)
    
    def generate_enhanced_document_embedding(self, content: str, entities: List[Dict], 
                                           relationships: List[Dict], metadata: Dict) -> np.ndarray:
        """
        Generate enhanced document embedding that incorporates entity context.
        
        Args:
            content: Raw document content
            entities: Extracted entities with their properties
            relationships: Inferred relationships
            metadata: Document metadata
            
        Returns:
            Enhanced embedding vector
        """
        # Base document embedding
        base_embedding = self.embedding_model.encode([content])[0]
        
        # Entity context embedding
        entity_context = self._create_entity_context(entities)
        entity_embedding = self.embedding_model.encode([entity_context])[0] if entity_context else np.zeros_like(base_embedding)
        
        # Relationship context embedding
        relationship_context = self._create_relationship_context(relationships)
        rel_embedding = self.embedding_model.encode([relationship_context])[0] if relationship_context else np.zeros_like(base_embedding)
        
        # Combine embeddings with weighted average
        # Base content gets highest weight, then entities, then relationships
        weights = [0.6, 0.25, 0.15]
        enhanced_embedding = (
            weights[0] * base_embedding + 
            weights[1] * entity_embedding + 
            weights[2] * rel_embedding
        )
        
        # Normalize the result
        enhanced_embedding = enhanced_embedding / np.linalg.norm(enhanced_embedding)
        
        return enhanced_embedding
    
    def generate_entity_embedding(self, entity: Dict, context: str = "") -> np.ndarray:
        """
        Generate entity-specific embedding incorporating entity properties and context.
        
        Args:
            entity: Entity dictionary with text, label, properties
            context: Surrounding text context
            
        Returns:
            Entity embedding vector
        """
        # Create rich entity description
        entity_desc = self._create_entity_description(entity, context)
        
        # Generate embedding
        embedding = self.embedding_model.encode([entity_desc])[0]
        
        return embedding
    
    def generate_relationship_embedding(self, relationship: Dict, source_entity: Dict, 
                                      target_entity: Dict, context: str = "") -> np.ndarray:
        """
        Generate relationship-specific embedding.
        
        Args:
            relationship: Relationship dictionary
            source_entity: Source entity
            target_entity: Target entity
            context: Context where relationship occurs
            
        Returns:
            Relationship embedding vector
        """
        # Create relationship description
        rel_desc = self._create_relationship_description(relationship, source_entity, target_entity, context)
        
        # Generate embedding
        embedding = self.embedding_model.encode([rel_desc])[0]
        
        return embedding
    
    def generate_hybrid_embedding(self, content: str, entities: List[Dict], 
                                 relationships: List[Dict], metadata: Dict) -> np.ndarray:
        """
        Generate hybrid embedding that combines multiple modalities.
        
        Args:
            content: Document content
            entities: Extracted entities
            relationships: Inferred relationships
            metadata: Document metadata
            
        Returns:
            Hybrid embedding vector
        """
        # Document embedding
        doc_embedding = self.embedding_model.encode([content])[0]
        
        # Entity cluster embedding (average of key entities)
        key_entities = [e for e in entities if e['label'] in ['PERSON', 'ORG', 'EVENT']]
        if key_entities:
            entity_embeddings = []
            for entity in key_entities[:5]:  # Top 5 key entities
                entity_emb = self.generate_entity_embedding(entity, content)
                entity_embeddings.append(entity_emb)
            entity_cluster = np.mean(entity_embeddings, axis=0)
        else:
            entity_cluster = np.zeros_like(doc_embedding)
        
        # Temporal embedding (encode time-based features)
        temporal_features = self._create_temporal_features(metadata)
        temporal_embedding = self.embedding_model.encode([temporal_features])[0] if temporal_features else np.zeros_like(doc_embedding)
        
        # Combine with sophisticated weighting
        weights = [0.5, 0.3, 0.2]  # doc, entities, temporal
        hybrid_embedding = (
            weights[0] * doc_embedding +
            weights[1] * entity_cluster +
            weights[2] * temporal_embedding
        )
        
        # Normalize
        hybrid_embedding = hybrid_embedding / np.linalg.norm(hybrid_embedding)
        
        return hybrid_embedding
    
    def index_enhanced_document(self, content: str, entities: List[Dict], 
                               relationships: List[Dict], metadata: Dict) -> bool:
        """
        Index document with multiple embedding types for enhanced search.
        
        Args:
            content: Document content
            entities: Extracted entities
            relationships: Inferred relationships
            metadata: Document metadata
            
        Returns:
            Success status
        """
        try:
            doc_id = metadata['content_hash']
            
            # Generate all embedding types
            enhanced_doc_emb = self.generate_enhanced_document_embedding(content, entities, relationships, metadata)
            hybrid_emb = self.generate_hybrid_embedding(content, entities, relationships, metadata)
            
            # Prepare metadata for ChromaDB
            chroma_metadata = self._prepare_metadata(metadata, entities, relationships)
            
            # Index enhanced document embedding
            self.collections['documents'].add(
                documents=[content],
                embeddings=[enhanced_doc_emb.tolist()],
                metadatas=[chroma_metadata],
                ids=[f"doc_{doc_id}"]
            )
            
            # Index hybrid embedding
            hybrid_context = self._create_hybrid_context(content, entities, relationships)
            self.collections['hybrid'].add(
                documents=[hybrid_context],
                embeddings=[hybrid_emb.tolist()],
                metadatas=[chroma_metadata],
                ids=[f"hybrid_{doc_id}"]
            )
            
            # Index individual entities
            for i, entity in enumerate(entities):
                entity_emb = self.generate_entity_embedding(entity, content)
                entity_metadata = {
                    **chroma_metadata,
                    'entity_text': entity['text'],
                    'entity_label': entity['label'],
                    'entity_canonical': entity.get('canonical_name', entity['text'])
                }
                
                self.collections['entities'].add(
                    documents=[self._create_entity_description(entity, content)],
                    embeddings=[entity_emb.tolist()],
                    metadatas=[entity_metadata],
                    ids=[f"entity_{doc_id}_{i}"]
                )
            
            # Index relationships
            for i, rel in enumerate(relationships):
                source_entity = next((e for e in entities if e['text'] == rel['source']), None)
                target_entity = next((e for e in entities if e['text'] == rel['target']), None)
                
                if source_entity and target_entity:
                    rel_emb = self.generate_relationship_embedding(rel, source_entity, target_entity, content)
                    rel_metadata = {
                        **chroma_metadata,
                        'relationship_type': rel['type'],
                        'source_entity': rel['source'],
                        'target_entity': rel['target']
                    }
                    
                    self.collections['relationships'].add(
                        documents=[self._create_relationship_description(rel, source_entity, target_entity, content)],
                        embeddings=[rel_emb.tolist()],
                        metadatas=[rel_metadata],
                        ids=[f"rel_{doc_id}_{i}"]
                    )
            
            logger.debug(f"Enhanced indexing complete for {metadata['file_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index enhanced embeddings for {metadata.get('file_name')}: {e}")
            return False
    
    def _create_entity_context(self, entities: List[Dict]) -> str:
        """Create entity context string for embedding."""
        if not entities:
            return ""
        
        # Group entities by type
        entity_groups = {}
        for entity in entities:
            label = entity['label']
            if label not in entity_groups:
                entity_groups[label] = []
            entity_groups[label].append(entity['text'])
        
        # Create context string
        context_parts = []
        for label, entity_list in entity_groups.items():
            unique_entities = list(set(entity_list))  # Remove duplicates
            context_parts.append(f"{label}: {', '.join(unique_entities[:5])}")  # Limit to top 5
        
        return " | ".join(context_parts)
    
    def _create_relationship_context(self, relationships: List[Dict]) -> str:
        """Create relationship context string for embedding."""
        if not relationships:
            return ""
        
        # Group relationships by type
        rel_groups = {}
        for rel in relationships:
            rel_type = rel['type']
            if rel_type not in rel_groups:
                rel_groups[rel_type] = []
            rel_groups[rel_type].append(f"{rel['source']} -> {rel['target']}")
        
        # Create context string
        context_parts = []
        for rel_type, rel_list in rel_groups.items():
            context_parts.append(f"{rel_type}: {'; '.join(rel_list[:3])}")  # Limit to top 3
        
        return " | ".join(context_parts)
    
    def _create_entity_description(self, entity: Dict, context: str = "") -> str:
        """Create rich entity description for embedding."""
        desc_parts = [f"Entity: {entity['text']}"]
        desc_parts.append(f"Type: {entity['label']}")
        
        if entity.get('canonical_name') and entity['canonical_name'] != entity['text']:
            desc_parts.append(f"Also known as: {entity['canonical_name']}")
        
        # Add context snippet
        if context and entity.get('start') and entity.get('end'):
            start = max(0, entity['start'] - 50)
            end = min(len(context), entity['end'] + 50)
            context_snippet = context[start:end].replace('\n', ' ').strip()
            desc_parts.append(f"Context: {context_snippet}")
        
        return " | ".join(desc_parts)
    
    def _create_relationship_description(self, relationship: Dict, source_entity: Dict, 
                                       target_entity: Dict, context: str = "") -> str:
        """Create relationship description for embedding."""
        desc = f"Relationship: {source_entity['text']} {relationship['type']} {target_entity['text']}"
        
        # Add relationship properties
        if relationship.get('properties'):
            props = []
            for key, value in relationship['properties'].items():
                if key not in ['source_file', 'confidence']:  # Skip internal props
                    props.append(f"{key}: {value}")
            if props:
                desc += f" | Properties: {', '.join(props)}"
        
        return desc
    
    def _create_temporal_features(self, metadata: Dict) -> str:
        """Create temporal feature description."""
        features = []
        
        if metadata.get('date'):
            features.append(f"Date: {metadata['date']}")
        
        if metadata.get('created_at'):
            try:
                dt = datetime.fromisoformat(metadata['created_at'])
                features.append(f"Created: {dt.strftime('%Y-%m-%d')}")
                features.append(f"Weekday: {dt.strftime('%A')}")
                features.append(f"Month: {dt.strftime('%B')}")
            except (ValueError, TypeError):
                pass
        
        return " | ".join(features) if features else ""
    
    def _create_hybrid_context(self, content: str, entities: List[Dict], relationships: List[Dict]) -> str:
        """Create hybrid context combining content, entities, and relationships."""
        # Content summary (first 200 chars)
        content_summary = content[:200].replace('\n', ' ').strip()
        
        # Key entities
        key_entities = [e['text'] for e in entities if e['label'] in ['PERSON', 'ORG', 'EVENT']][:5]
        
        # Key relationships
        key_rels = [f"{r['source']} {r['type']} {r['target']}" for r in relationships[:3]]
        
        parts = [f"Content: {content_summary}"]
        if key_entities:
            parts.append(f"Key Entities: {', '.join(key_entities)}")
        if key_rels:
            parts.append(f"Relationships: {'; '.join(key_rels)}")
        
        return " | ".join(parts)
    
    def _prepare_metadata(self, metadata: Dict, entities: List[Dict], relationships: List[Dict]) -> Dict:
        """Prepare metadata for ChromaDB storage."""
        # Convert datetime strings to timestamps for temporal filtering
        chroma_metadata = {}
        
        # Basic metadata
        for key in ['file_name', 'title', 'file_size']:
            if key in metadata:
                chroma_metadata[key] = metadata[key]
        
        # Date handling
        if metadata.get('date'):
            chroma_metadata['date'] = metadata['date']
        
        # Timestamp conversion
        for time_field in ['created_at', 'modified_at']:
            if metadata.get(time_field):
                try:
                    dt = datetime.fromisoformat(metadata[time_field])
                    chroma_metadata[f"{time_field}_ts"] = dt.timestamp()
                    chroma_metadata[f"{time_field}_iso"] = metadata[time_field]
                except (ValueError, TypeError):
                    pass

        # Ensure created_at_ts exists for temporal filtering
        if 'created_at_ts' not in chroma_metadata:
            ts_source = (
                metadata.get('date') or
                metadata.get('created_at_iso') or
                metadata.get('created_at')
            )
            if ts_source:
                try:
                    dt = datetime.fromisoformat(ts_source[:26])  # trim possible Z / offset for fromisoformat
                    chroma_metadata['created_at_ts'] = dt.timestamp()
                    chroma_metadata.setdefault('created_at_iso', dt.isoformat())
                except (ValueError, TypeError):
                    logger.warning(f"⚠️ Unable to derive created_at_ts from '{ts_source}'")

        # Entity and relationship counts
        chroma_metadata['entity_count'] = len(entities)
        chroma_metadata['relationship_count'] = len(relationships)
        
        # Entity type distribution
        entity_types = {}
        for entity in entities:
            label = entity['label']
            entity_types[label] = entity_types.get(label, 0) + 1
        
        for entity_type, count in entity_types.items():
            chroma_metadata[f"entities_{entity_type.lower()}"] = count
        
        return chroma_metadata


class EnhancedQueryInterface:
    """
    Enhanced query interface that leverages multiple embedding types
    for sophisticated search and retrieval.
    """
    
    def __init__(self, embedding_generator: EnhancedEmbeddingGenerator):
        self.embedding_gen = embedding_generator
        self.collections = embedding_generator.collections
        self.embedding_model = embedding_generator.embedding_model
    
    def hybrid_search(self, query: str, search_types: List[str] = None, 
                     filters: Dict = None, limit: int = 10) -> List[Dict]:
        """
        Perform hybrid search across multiple embedding types.
        
        Args:
            query: Search query
            search_types: Types to search ['documents', 'entities', 'relationships', 'hybrid']
            filters: Metadata filters
            limit: Maximum results
            
        Returns:
            Ranked search results
        """
        if search_types is None:
            search_types = ['documents', 'entities', 'relationships']
        
        query_embedding = self.embedding_model.encode([query])[0]
        
        all_results = []
        
        for search_type in search_types:
            if search_type in self.collections:
                try:
                    results = self.collections[search_type].query(
                        query_embeddings=[query_embedding.tolist()],
                        n_results=limit,
                        where=filters
                    )
                    
                    # Add search type to results
                    for i, doc in enumerate(results['documents'][0]):
                        result = {
                            'content': doc,
                            'metadata': results['metadatas'][0][i],
                            'distance': results['distances'][0][i],
                            'search_type': search_type,
                            'score': 1 - results['distances'][0][i]  # Convert distance to similarity score
                        }
                        all_results.append(result)
                        
                except Exception as e:
                    logger.error(f"Error searching {search_type}: {e}")
        
        # Sort by score and return top results
        all_results.sort(key=lambda x: x['score'], reverse=True)
        return all_results[:limit]
    
    def entity_centric_search(self, entity_name: str, entity_type: str = None, 
                             limit: int = 10) -> List[Dict]:
        """
        Search for documents related to a specific entity.
        
        Args:
            entity_name: Name of the entity
            entity_type: Optional entity type filter
            limit: Maximum results
            
        Returns:
            Entity-related documents
        """
        filters = {'entity_text': entity_name}
        if entity_type:
            filters['entity_label'] = entity_type
        
        # Search entity embeddings
        query_embedding = self.embedding_model.encode([f"Entity: {entity_name}"])[0]
        
        results = self.collections['entities'].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit,
            where=filters
        )
        
        return self._format_results(results, 'entity_search')
    
    def relationship_search(self, source_entity: str, target_entity: str = None, 
                           relationship_type: str = None, limit: int = 10) -> List[Dict]:
        """
        Search for relationships involving specific entities.
        
        Args:
            source_entity: Source entity name
            target_entity: Optional target entity name
            relationship_type: Optional relationship type
            limit: Maximum results
            
        Returns:
            Relationship results
        """
        # Build query
        query_parts = [f"Relationship: {source_entity}"]
        if target_entity:
            query_parts.append(target_entity)
        if relationship_type:
            query_parts.append(relationship_type)
        
        query = " ".join(query_parts)
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Build filters
        filters = {'source_entity': source_entity}
        if target_entity:
            filters['target_entity'] = target_entity
        if relationship_type:
            filters['relationship_type'] = relationship_type
        
        results = self.collections['relationships'].query(
            query_embeddings=[query_embedding.tolist()],
            n_results=limit,
            where=filters
        )
        
        return self._format_results(results, 'relationship_search')
    
    def temporal_search(self, query: str, start_date: str = None, end_date: str = None, 
                       search_type: str = 'documents', limit: int = 10) -> List[Dict]:
        """
        Search with temporal constraints.
        
        Args:
            query: Search query
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            search_type: Type of search ('documents', 'hybrid')
            limit: Maximum results
            
        Returns:
            Temporally filtered results
        """
        logger.info(f"🔍 temporal_search method called:")
        logger.info(f"   - query: {query}")
        logger.info(f"   - start_date: {start_date}")
        logger.info(f"   - end_date: {end_date}")
        logger.info(f"   - search_type: {search_type}")
        logger.info(f"   - limit: {limit}")
        
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            logger.info(f"✅ Query embedding generated: shape {query_embedding.shape}")
        except Exception as embed_e:
            logger.error(f"❌ Query embedding failed: {embed_e}")
            raise
        
        # Build temporal filters - ChromaDB supports $and for multiple conditions
        filters = {}
        if start_date and end_date:
            try:
                start_ts = datetime.fromisoformat(start_date).timestamp()
                end_dt = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, microsecond=999999)
                end_ts = end_dt.timestamp()
                logger.info(f"📅 Date range timestamps: {start_ts} to {end_ts}")
                filters = {
                    "$and": [
                        {"created_at_ts": {"$gte": start_ts}},
                        {"created_at_ts": {"$lte": end_ts}}
                    ]
                }
                logger.info(f"🔧 Using ChromaDB $and filter for date range: {filters}")
            except ValueError as date_e:
                logger.warning(f"⚠️ Invalid date format: {start_date} or {end_date} - {date_e}")
        elif start_date:
            try:
                start_ts = datetime.fromisoformat(start_date).timestamp()
                filters['created_at_ts'] = {'$gte': start_ts}
                logger.info(f"📅 Start date filter: created_at_ts >= {start_ts}")
            except ValueError:
                logger.warning(f"⚠️ Invalid start_date format: {start_date}")
        elif end_date:
            try:
                end_dt = datetime.fromisoformat(end_date).replace(hour=23, minute=59, second=59, microsecond=999999)
                end_ts = end_dt.timestamp()
                filters['created_at_ts'] = {'$lte': end_ts}
                logger.info(f"📅 End date filter: created_at_ts <= {end_ts}")
            except ValueError:
                logger.warning(f"⚠️ Invalid end_date format: {end_date}")
        
        if search_type not in self.collections:
            logger.warning(f"⚠️ Invalid search_type '{search_type}', defaulting to 'documents'")
            search_type = 'documents'
        
        collection = self.collections[search_type]
        logger.info(f"📚 Using collection '{search_type}' with {collection.count()} documents")
        logger.info(f"🔍 ChromaDB query parameters: n_results={limit}, filters={filters}")
        
        try:
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=limit,
                where=filters if filters else None
            )
            logger.info(f"✅ ChromaDB query completed:")
            logger.info(f"   - documents: {len(results.get('documents', []))} batches")
            if results.get('documents'):
                logger.info(f"   - documents[0]: {len(results['documents'][0])} items")
            logger.info(f"   - metadatas: {len(results.get('metadatas', []))} batches")
            if results.get('metadatas'):
                logger.info(f"   - metadatas[0]: {len(results['metadatas'][0])} items")
        except Exception as query_e:
            logger.error(f"❌ ChromaDB query failed: {query_e}")
            raise
        
        # Date filtering in Python is no longer the primary method for range queries
        # but can serve as a fallback or for complex logic not supported by ChromaDB
        # The main filtering is now done in the 'where' clause.
        
        logger.info(f"📝 Formatting {len(results.get('documents', [[]])[0])} results...")
        formatted_results = self._format_results(results, 'temporal_search')
        logger.info(f"✅ temporal_search completed: returning {len(formatted_results)} formatted results")
        
        return formatted_results
    
    def _format_results(self, results: Dict, search_type: str) -> List[Dict]:
        """Format search results consistently."""
        formatted = []
        
        if results['documents'] and results['documents'][0]:
            for i, doc in enumerate(results['documents'][0]):
                result = {
                    'content': doc,
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i],
                    'score': 1 - results['distances'][0][i],
                    'search_type': search_type
                }
                formatted.append(result)
        
        return formatted


# Factory function for easy import
def create_enhanced_embedding_system(db_manager):
    """Create an enhanced embedding system with generator and query interface."""
    generator = EnhancedEmbeddingGenerator(db_manager)
    query_interface = EnhancedQueryInterface(generator)
    
    return generator, query_interface 