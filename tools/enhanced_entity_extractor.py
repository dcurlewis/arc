"""
Enhanced Entity Extractor with Custom spaCy Configuration
Provides context-aware entity extraction, disambiguation, and relationship inference.
"""

import re
import spacy
from spacy import displacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Span
from spacy.lang.en import English
from typing import Dict, List, Any, Set, Tuple, Optional
import logging
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)


class EnhancedEntityExtractor:
    """Enhanced entity extractor with custom spaCy configuration and disambiguation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nlp = None
        self.matcher = None
        self.phrase_matcher = None
        self.entity_patterns = {}
        self.disambiguation_rules = {}
        self.entity_aliases = {}
        self.domain_entities = {}
        self._setup_nlp_pipeline()
        self._load_custom_patterns()
        self._load_disambiguation_rules()
    
    def _setup_nlp_pipeline(self):
        """Set up enhanced spaCy pipeline with custom components."""
        try:
            model_name = self.config.get('spacy.model', 'en_core_web_lg')
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded base spaCy model: {model_name}")
            
            # Add custom pipeline components
            self._add_custom_entity_ruler()
            self._add_meeting_detector()
            self._add_organization_enhancer()
            self._add_person_title_detector()
            self._add_date_normalizer()
            
            logger.info("Enhanced spaCy pipeline configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to setup enhanced spaCy pipeline: {e}")
            raise
    
    def _add_custom_entity_ruler(self):
        """Add custom entity ruler for domain-specific entities."""
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
            
            # Define custom patterns for business/tech entities
            patterns = [
                # Meeting types
                {"label": "MEETING_TYPE", "pattern": [{"LOWER": {"IN": ["standup", "retrospective", "planning", "demo", "review"]}}]},
                {"label": "MEETING_TYPE", "pattern": [{"LOWER": "sprint"}, {"LOWER": {"IN": ["planning", "review", "retrospective"]}}]},
                {"label": "MEETING_TYPE", "pattern": [{"LOWER": "one"}, {"LOWER": "on"}, {"LOWER": "one"}]},
                {"label": "MEETING_TYPE", "pattern": [{"LOWER": "all"}, {"LOWER": "hands"}]},
                {"label": "MEETING_TYPE", "pattern": [{"LOWER": "town"}, {"LOWER": "hall"}]},
                
                # Project/Product names (common patterns)
                {"label": "PRODUCT", "pattern": [{"IS_TITLE": True}, {"LOWER": {"IN": ["platform", "system", "app", "tool", "service"]}}]},
                {"label": "PRODUCT", "pattern": [{"LOWER": {"IN": ["api", "sdk", "cli", "ui", "ux"]}}]},
                
                # Roles and titles
                {"label": "JOB_TITLE", "pattern": [{"LOWER": {"IN": ["ceo", "cto", "cfo", "vp", "director", "manager", "lead", "senior", "principal", "staff"]}}]},
                {"label": "JOB_TITLE", "pattern": [{"LOWER": "product"}, {"LOWER": "manager"}]},
                {"label": "JOB_TITLE", "pattern": [{"LOWER": "engineering"}, {"LOWER": "manager"}]},
                {"label": "JOB_TITLE", "pattern": [{"LOWER": "tech"}, {"LOWER": "lead"}]},
                {"label": "JOB_TITLE", "pattern": [{"LOWER": "software"}, {"LOWER": {"IN": ["engineer", "developer", "architect"]}}]},
                
                # Technology terms
                {"label": "TECHNOLOGY", "pattern": [{"LOWER": {"IN": ["python", "javascript", "react", "nodejs", "docker", "kubernetes", "aws", "gcp", "azure"]}}]},
                {"label": "TECHNOLOGY", "pattern": [{"LOWER": "machine"}, {"LOWER": "learning"}]},
                {"label": "TECHNOLOGY", "pattern": [{"LOWER": "artificial"}, {"LOWER": "intelligence"}]},
                
                # Time periods
                {"label": "TIME_PERIOD", "pattern": [{"LOWER": {"IN": ["q1", "q2", "q3", "q4"]}}]},
                {"label": "TIME_PERIOD", "pattern": [{"LOWER": "quarter"}, {"SHAPE": "d"}]},
                {"label": "TIME_PERIOD", "pattern": [{"LOWER": {"IN": ["sprint", "iteration"]}}, {"SHAPE": "d"}]},
            ]
            
            ruler.add_patterns(patterns)
            logger.info(f"Added {len(patterns)} custom entity patterns")
    
    def _add_meeting_detector(self):
        """Add custom component to detect and enhance meeting-related entities."""
        @spacy.Language.component("meeting_detector")
        def meeting_detector(doc):
            # Look for meeting indicators
            meeting_indicators = ["meeting", "call", "sync", "standup", "retrospective", "planning"]
            
            for token in doc:
                if token.text.lower() in meeting_indicators:
                    # Look for participants mentioned nearby
                    window_start = max(0, token.i - 10)
                    window_end = min(len(doc), token.i + 10)
                    
                    for i in range(window_start, window_end):
                        if doc[i].ent_type_ == "PERSON":
                            # Tag as meeting participant
                            doc[i]._.is_meeting_participant = True
            
            return doc
        
        # Add custom attributes
        spacy.tokens.Token.set_extension("is_meeting_participant", default=False, force=True)
        
        if "meeting_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("meeting_detector", after="ner")
    
    def _add_organization_enhancer(self):
        """Add component to enhance organization detection."""
        @spacy.Language.component("organization_enhancer")
        def organization_enhancer(doc):
            # Common organization suffixes
            org_suffixes = ["inc", "corp", "llc", "ltd", "co", "company", "corporation"]
            
            for token in doc:
                if token.text.lower() in org_suffixes and token.i > 0:
                    # Check if previous token could be part of org name
                    prev_token = doc[token.i - 1]
                    if prev_token.is_title and not prev_token.ent_type_:
                        # Create organization entity spanning both tokens
                        span = Span(doc, prev_token.i, token.i + 1, label="ORG")
                        doc.ents = list(doc.ents) + [span]
            
            return doc
        
        if "organization_enhancer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("organization_enhancer", after="ner")
    
    def _add_person_title_detector(self):
        """Add component to detect person titles and enhance person entities."""
        @spacy.Language.component("person_title_detector")
        def person_title_detector(doc):
            titles = ["mr", "mrs", "ms", "dr", "prof", "professor"]
            
            for token in doc:
                if token.text.lower() in titles and token.i < len(doc) - 1:
                    next_token = doc[token.i + 1]
                    if next_token.ent_type_ == "PERSON":
                        # Extend person entity to include title
                        for ent in doc.ents:
                            if ent.start == next_token.i:
                                new_span = Span(doc, token.i, ent.end, label="PERSON")
                                doc.ents = [new_span if e == ent else e for e in doc.ents]
                                break
            
            return doc
        
        if "person_title_detector" not in self.nlp.pipe_names:
            self.nlp.add_pipe("person_title_detector", after="organization_enhancer")
    
    def _add_date_normalizer(self):
        """Add component to normalize and enhance date entities."""
        @spacy.Language.component("date_normalizer")
        def date_normalizer(doc):
            # Add custom date patterns
            date_patterns = [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',    # YYYY/MM/DD or YYYY-MM-DD
                r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}[,\s]+\d{4}\b',  # Month DD, YYYY
                r'\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{4}\b',  # DD Month YYYY
            ]
            
            text = doc.text
            for pattern in date_patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    start_char = match.start()
                    end_char = match.end()
                    
                    # Find token span for this character range
                    start_token = None
                    end_token = None
                    for token in doc:
                        if token.idx <= start_char < token.idx + len(token.text):
                            start_token = token.i
                        if token.idx < end_char <= token.idx + len(token.text):
                            end_token = token.i + 1
                            break
                    
                    if start_token is not None and end_token is not None:
                        # Check for overlaps before adding
                        new_span = Span(doc, start_token, end_token, label="DATE")
                        overlaps = False
                        for existing_ent in doc.ents:
                            if (new_span.start < existing_ent.end and new_span.end > existing_ent.start):
                                overlaps = True
                                break
                        
                        if not overlaps:
                            doc.ents = list(doc.ents) + [new_span]
            
            return doc
        
        if "date_normalizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("date_normalizer", after="person_title_detector")
    
    def _load_custom_patterns(self):
        """Load custom entity patterns from configuration."""
        self.matcher = Matcher(self.nlp.vocab)
        self.phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        
        # Add some basic token patterns to avoid warnings
        basic_patterns = [
            ("TECHNOLOGY", [{"LOWER": {"IN": ["python", "javascript", "react", "docker", "kubernetes"]}}]),
            ("JOB_TITLE", [{"LOWER": "software"}, {"LOWER": "engineer"}]),
            ("MEETING_TYPE", [{"LOWER": "one"}, {"LOWER": "on"}, {"LOWER": "one"}])
        ]
        
        for label, pattern in basic_patterns:
            self.matcher.add(label, [pattern])
        
        # Load patterns from config
        entity_config = self.config.get('entity', {})
        patterns_config = entity_config.get('custom_patterns', {})
        
        for entity_type, patterns in patterns_config.items():
            if isinstance(patterns, list):
                # Add phrase patterns
                phrases = [self.nlp(pattern) for pattern in patterns]
                self.phrase_matcher.add(entity_type, phrases)
                logger.debug(f"Added {len(patterns)} phrase patterns for {entity_type}")
    
    def _load_disambiguation_rules(self):
        """Load entity disambiguation rules from configuration."""
        # Load from config structure
        entity_config = self.config.get('entity', {})
        
        # Basic disambiguation rules
        self.disambiguation_rules = entity_config.get('disambiguation_rules', {})
        
        # Entity aliases (different names for same entity)
        self.entity_aliases = entity_config.get('aliases', {})
        
        # Domain-specific entity mappings
        self.domain_entities = entity_config.get('domain_mappings', {})
        
        logger.info(f"Loaded {len(self.disambiguation_rules)} disambiguation rules")
        logger.info(f"Loaded {len(self.entity_aliases)} entity aliases")
        logger.info(f"Loaded {len(self.domain_entities)} domain entity mappings")
    
    def extract_entities(self, text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Extract entities with enhanced processing and disambiguation."""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        # Extract standard entities
        for ent in doc.ents:
            if len(ent.text.strip()) < 2:
                continue
            
            entity = {
                'text': ent.text.strip(),
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 1.0)
            }
            
            # Apply disambiguation
            entity = self._apply_enhanced_disambiguation(entity, doc, context)
            
            # Add context-specific information
            entity = self._add_context_information(entity, doc, context)
            
            entities.append(entity)
        
        # Extract custom pattern matches
        custom_entities = self._extract_custom_pattern_entities(doc, context)
        entities.extend(custom_entities)
        
        # Deduplicate and merge similar entities
        entities = self._intelligent_deduplication(entities)
        
        # Apply confidence scoring
        entities = self._apply_confidence_scoring(entities, text, context)
        
        return entities
    
    def _apply_enhanced_disambiguation(self, entity: Dict[str, Any], doc, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Apply enhanced disambiguation rules."""
        text = entity['text']
        original_text = text
        
        # Get the rules (now already properly loaded)
        disambiguation_rules = self.disambiguation_rules
        entity_aliases = self.entity_aliases
        domain_entities = self.domain_entities
        
        # Apply basic disambiguation rules
        if text in disambiguation_rules:
            entity['canonical_name'] = disambiguation_rules[text]
            entity['disambiguation_rule'] = 'explicit_mapping'
        
        # Apply alias resolution
        elif text in entity_aliases:
            entity['canonical_name'] = entity_aliases[text]
            entity['disambiguation_rule'] = 'alias_resolution'
        
        # Apply domain-specific disambiguation
        elif entity['label'] in domain_entities:
            domain_map = domain_entities[entity['label']]
            text_lower = text.lower()
            for pattern, canonical in domain_map.items():
                if pattern.lower() in text_lower:
                    entity['canonical_name'] = canonical
                    entity['disambiguation_rule'] = 'domain_pattern'
                    break
        
        # Context-based disambiguation
        if context:
            entity = self._apply_context_disambiguation(entity, context)
        
        # If no canonical name set, use original text
        if 'canonical_name' not in entity:
            entity['canonical_name'] = original_text
        
        return entity
    
    def _apply_context_disambiguation(self, entity: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply context-based disambiguation rules."""
        # File name context
        file_name = context.get('file_name', '')
        if file_name:
            # If entity appears in filename, boost confidence
            if entity['text'].lower() in file_name.lower():
                entity['confidence'] = min(1.0, entity.get('confidence', 1.0) + 0.2)
                entity['context_boost'] = 'filename_match'
        
        # Date context
        file_date = context.get('date')
        if file_date and entity['label'] == 'DATE':
            # If date entity matches file date, high confidence
            try:
                if file_date in entity['text']:
                    entity['confidence'] = 0.95
                    entity['context_boost'] = 'file_date_match'
            except:
                pass
        
        # Meeting context
        title = context.get('title', '').lower()
        if 'meeting' in title or 'call' in title:
            if entity['label'] == 'PERSON':
                entity['meeting_participant'] = True
                entity['confidence'] = min(1.0, entity.get('confidence', 1.0) + 0.1)
        
        return entity
    
    def _add_context_information(self, entity: Dict[str, Any], doc, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Add contextual information to entities."""
        # Add surrounding context
        start_token = None
        end_token = None
        
        for token in doc:
            if token.idx <= entity['start'] < token.idx + len(token.text):
                start_token = token.i
            if token.idx < entity['end'] <= token.idx + len(token.text):
                end_token = token.i
                break
        
        if start_token is not None and end_token is not None:
            # Get surrounding tokens for context
            context_start = max(0, start_token - 3)
            context_end = min(len(doc), end_token + 3)
            
            before_context = [t.text for t in doc[context_start:start_token]]
            after_context = [t.text for t in doc[end_token:context_end]]
            
            entity['context_before'] = ' '.join(before_context)
            entity['context_after'] = ' '.join(after_context)
            
            # Check for title/role indicators
            if entity['label'] == 'PERSON':
                entity = self._detect_person_roles(entity, doc, start_token, end_token)
        
        return entity
    
    def _detect_person_roles(self, entity: Dict[str, Any], doc, start_token: int, end_token: int) -> Dict[str, Any]:
        """Detect roles/titles for person entities."""
        role_indicators = ['ceo', 'cto', 'manager', 'director', 'lead', 'engineer', 'developer']
        
        # Look for roles in surrounding context
        context_window = 5
        start_search = max(0, start_token - context_window)
        end_search = min(len(doc), end_token + context_window)
        
        roles = []
        for i in range(start_search, end_search):
            token = doc[i]
            if token.text.lower() in role_indicators:
                roles.append(token.text.lower())
            elif token.ent_type_ == 'JOB_TITLE':
                roles.append(token.text)
        
        if roles:
            entity['roles'] = list(set(roles))
        
        return entity
    
    def _extract_custom_pattern_entities(self, doc, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Extract entities using custom patterns."""
        entities = []
        
        # Use phrase matcher
        matches = self.phrase_matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            entity = {
                'text': span.text,
                'label': label,
                'start': span.start_char,
                'end': span.end_char,
                'confidence': 0.8,  # Custom patterns get medium confidence
                'extraction_method': 'custom_pattern'
            }
            
            entities.append(entity)
        
        # Use token matcher
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            label = self.nlp.vocab.strings[match_id]
            span = doc[start:end]
            
            entity = {
                'text': span.text,
                'label': label,
                'start': span.start_char,
                'end': span.end_char,
                'confidence': 0.8,
                'extraction_method': 'token_pattern'
            }
            
            entities.append(entity)
        
        return entities
    
    def _intelligent_deduplication(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Intelligent deduplication with entity merging."""
        # Group by canonical name
        groups = defaultdict(list)
        for entity in entities:
            canonical = entity.get('canonical_name', entity['text'])
            groups[canonical.lower()].append(entity)
        
        deduplicated = []
        for canonical, group in groups.items():
            if len(group) == 1:
                deduplicated.append(group[0])
            else:
                # Merge group into best representative
                best = max(group, key=lambda e: (
                    e.get('confidence', 0),
                    len(e['text']),
                    1 if e.get('disambiguation_rule') else 0
                ))
                
                # Merge information from other entities
                best['alternative_forms'] = list(set([e['text'] for e in group if e['text'] != best['text']]))
                best['mention_count'] = len(group)
                
                deduplicated.append(best)
        
        return deduplicated
    
    def _apply_confidence_scoring(self, entities: List[Dict[str, Any]], text: str, context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Apply sophisticated confidence scoring."""
        for entity in entities:
            base_confidence = entity.get('confidence', 1.0)
            
            # Boost confidence for entities with disambiguation rules
            if entity.get('disambiguation_rule'):
                base_confidence = min(1.0, base_confidence + 0.1)
            
            # Boost confidence for entities mentioned multiple times
            mention_count = entity.get('mention_count', 1)
            if mention_count > 1:
                base_confidence = min(1.0, base_confidence + (mention_count - 1) * 0.05)
            
            # Reduce confidence for very short entities
            if len(entity['text']) <= 2:
                base_confidence *= 0.7
            
            # Boost confidence for title case entities (likely proper nouns)
            if entity['text'].istitle():
                base_confidence = min(1.0, base_confidence + 0.05)
            
            entity['confidence'] = round(base_confidence, 3)
        
        return entities
    
    def infer_relationships(self, entities: List[Dict[str, Any]], text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer relationships with enhanced logic."""
        relationships = []
        
        # Meeting relationships
        relationships.extend(self._infer_meeting_relationships(entities, text, metadata))
        
        # Organization relationships
        relationships.extend(self._infer_organization_relationships(entities, text, metadata))
        
        # Project relationships
        relationships.extend(self._infer_project_relationships(entities, text, metadata))
        
        # Temporal relationships
        relationships.extend(self._infer_temporal_relationships(entities, text, metadata))
        
        return relationships
    
    def _infer_meeting_relationships(self, entities: List[Dict[str, Any]], text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer meeting-related relationships."""
        relationships = []
        
        # Check if this is meeting content
        is_meeting = (
            'meeting' in metadata.get('title', '').lower() or
            'call' in metadata.get('title', '').lower() or
            any('meeting' in e.get('context_before', '').lower() + e.get('context_after', '').lower() 
                for e in entities if e['label'] == 'PERSON')
        )
        
        if is_meeting:
            people = [e for e in entities if e['label'] == 'PERSON']
            for i, person1 in enumerate(people):
                for person2 in people[i+1:]:
                    relationships.append({
                        'source': person1.get('canonical_name', person1['text']),
                        'target': person2.get('canonical_name', person2['text']),
                        'type': 'ATTENDED_MEETING_WITH',
                        'properties': {
                            'meeting_date': metadata.get('date'),
                            'meeting_title': metadata.get('title'),
                            'confidence': min(person1.get('confidence', 1.0), person2.get('confidence', 1.0)),
                            'created_at': datetime.now().isoformat()
                        }
                    })
        
        return relationships
    
    def _infer_organization_relationships(self, entities: List[Dict[str, Any]], text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer organization-related relationships."""
        relationships = []
        
        people = [e for e in entities if e['label'] == 'PERSON']
        orgs = [e for e in entities if e['label'] == 'ORG']
        
        for person in people:
            for org in orgs:
                # Enhanced proximity and context analysis
                confidence = self._calculate_relationship_confidence(person, org, text)
                
                if confidence > 0.5:  # Only create relationship if reasonably confident
                    relationships.append({
                        'source': person.get('canonical_name', person['text']),
                        'target': org.get('canonical_name', org['text']),
                        'type': 'AFFILIATED_WITH',
                        'properties': {
                            'source_file': metadata.get('file_name'),
                            'confidence': confidence,
                            'created_at': datetime.now().isoformat()
                        }
                    })
        
        return relationships
    
    def _calculate_relationship_confidence(self, entity1: Dict[str, Any], entity2: Dict[str, Any], text: str) -> float:
        """Calculate confidence for a potential relationship."""
        # Distance-based confidence
        distance = abs(entity1['start'] - entity2['start'])
        max_distance = 300  # characters
        distance_confidence = max(0, 1 - (distance / max_distance))
        
        # Context keyword confidence
        start = min(entity1['start'], entity2['start'])
        end = max(entity1['end'], entity2['end'])
        context = text[start:end].lower()
        
        relationship_keywords = [
            'works at', 'employed by', 'from', 'at', 'with', 'joins', 'hired by',
            'manager at', 'engineer at', 'developer at', 'ceo of', 'founder of'
        ]
        
        keyword_confidence = 0
        for keyword in relationship_keywords:
            if keyword in context:
                keyword_confidence = 0.8
                break
        
        # Combine confidences
        base_confidence = max(distance_confidence, keyword_confidence)
        
        # Boost if both entities have high individual confidence
        entity_confidence = (entity1.get('confidence', 1.0) + entity2.get('confidence', 1.0)) / 2
        
        return min(1.0, base_confidence * entity_confidence)
    
    def _infer_project_relationships(self, entities: List[Dict[str, Any]], text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer project and product relationships."""
        relationships = []
        
        people = [e for e in entities if e['label'] == 'PERSON']
        products = [e for e in entities if e['label'] in ['PRODUCT', 'PROJECT']]
        
        for person in people:
            for product in products:
                confidence = self._calculate_relationship_confidence(person, product, text)
                
                if confidence > 0.4:  # Lower threshold for project relationships
                    relationships.append({
                        'source': person.get('canonical_name', person['text']),
                        'target': product.get('canonical_name', product['text']),
                        'type': 'WORKS_ON',
                        'properties': {
                            'source_file': metadata.get('file_name'),
                            'confidence': confidence,
                            'created_at': datetime.now().isoformat()
                        }
                    })
        
        return relationships
    
    def _infer_temporal_relationships(self, entities: List[Dict[str, Any]], text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Infer temporal relationships."""
        relationships = []
        
        dates = [e for e in entities if e['label'] == 'DATE']
        events = [e for e in entities if e['label'] in ['MEETING_TYPE', 'EVENT']]
        
        for date in dates:
            for event in events:
                confidence = self._calculate_relationship_confidence(date, event, text)
                
                if confidence > 0.6:
                    relationships.append({
                        'source': event.get('canonical_name', event['text']),
                        'target': date.get('canonical_name', date['text']),
                        'type': 'SCHEDULED_FOR',
                        'properties': {
                            'source_file': metadata.get('file_name'),
                            'confidence': confidence,
                            'created_at': datetime.now().isoformat()
                        }
                    })
        
        return relationships
    
    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get statistics about the extraction process."""
        return {
            'pipeline_components': list(self.nlp.pipe_names),
            'disambiguation_rules': len(self.disambiguation_rules),
            'entity_aliases': len(self.entity_aliases),
            'domain_mappings': len(self.domain_entities),
            'custom_patterns': len(self.phrase_matcher) if self.phrase_matcher else 0
        } 