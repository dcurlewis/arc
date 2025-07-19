#!/usr/bin/env python3
"""
Test script for Enhanced Entity Extractor
Tests the new spaCy configuration and custom disambiguation rules.
"""

import sys
import yaml
from pathlib import Path

# Add tools directory to path
sys.path.append(str(Path(__file__).parent))

from enhanced_entity_extractor import EnhancedEntityExtractor


def load_test_config():
    """Load the enhanced entity configuration."""
    config_path = Path("../config/enhanced_entity_config.yaml")
    template_path = Path("../config/enhanced_entity_config.template.yaml")
    
    # Try actual config first
    if config_path.exists():
        with open(config_path, 'r') as f:
            print(f"✅ Loaded config from {config_path}")
            return yaml.safe_load(f)
    
    # Fallback to template
    elif template_path.exists():
        with open(template_path, 'r') as f:
            print(f"📋 Using template config from {template_path}")
            print("💡 Consider copying the template to enhanced_entity_config.yaml and customizing it")
            return yaml.safe_load(f)
    
    # Basic fallback config
    else:
        print(f"⚠️  No config files found, using minimal configuration")
        return {
            'spacy': {'model': 'en_core_web_lg'},
            'entity': {
                'disambiguation_rules': {'Dave': 'David'},
                'aliases': {},
                'domain_mappings': {},
                'custom_patterns': {}
            }
        }


def test_basic_extraction():
    """Test basic entity extraction."""
    print("\n🧪 Testing Basic Entity Extraction...")
    
    config = load_test_config()
    extractor = EnhancedEntityExtractor(config)
    
    test_text = """
    Dave met with Yolanda from Google and Glen from Microsoft yesterday. 
    They discussed the AI platform project and scheduled a sprint planning meeting for next week.
    The engineering manager Ben will join the standup on Monday.
    """
    
    context = {
        'file_name': 'meeting-notes.md',
        'title': 'Team Meeting Notes',
        'date': '2025-07-19'
    }
    
    entities = extractor.extract_entities(test_text, context)
    
    print(f"📊 Extracted {len(entities)} entities:")
    for entity in entities:
        canonical = entity.get('canonical_name', entity['text'])
        confidence = entity.get('confidence', 1.0)
        disambiguation = entity.get('disambiguation_rule', 'none')
        
        print(f"  • {entity['text']} ({entity['label']}) → {canonical}")
        print(f"    Confidence: {confidence:.2f}, Rule: {disambiguation}")
    
    return entities


def test_relationship_inference():
    """Test relationship inference."""
    print("\n🔗 Testing Relationship Inference...")
    
    config = load_test_config()
    extractor = EnhancedEntityExtractor(config)
    
    test_text = """
    During the sprint planning meeting, Dave from the Engineering team worked with 
    Yolanda Li, who is a Product Manager at Google. Glen Pink, a senior engineer 
    at Microsoft, also attended the meeting via Zoom.
    """
    
    context = {
        'file_name': 'sprint-planning-notes.md',
        'title': 'Sprint Planning Meeting',
        'date': '2025-07-19'
    }
    
    entities = extractor.extract_entities(test_text, context)
    relationships = extractor.infer_relationships(entities, test_text, context)
    
    print(f"📊 Found {len(relationships)} relationships:")
    for rel in relationships:
        print(f"  • {rel['source']} --({rel['type']})-> {rel['target']}")
        print(f"    Confidence: {rel['properties'].get('confidence', 'N/A')}")
    
    return relationships


def test_custom_patterns():
    """Test custom pattern matching."""
    print("\n🎯 Testing Custom Pattern Matching...")
    
    config = load_test_config()
    extractor = EnhancedEntityExtractor(config)
    
    test_text = """
    The team used Django and React for the web platform. 
    The CTO scheduled a one on one with the senior software engineer.
    During Q3, we'll migrate to Kubernetes and implement CI/CD pipelines.
    """
    
    entities = extractor.extract_entities(test_text)
    
    custom_entities = [e for e in entities if e.get('extraction_method') in ['custom_pattern', 'token_pattern']]
    
    print(f"📊 Found {len(custom_entities)} custom pattern entities:")
    for entity in custom_entities:
        print(f"  • {entity['text']} ({entity['label']}) - {entity.get('extraction_method')}")
    
    return custom_entities


def test_disambiguation():
    """Test disambiguation rules."""
    print("\n🎭 Testing Disambiguation Rules...")
    
    config = load_test_config()
    extractor = EnhancedEntityExtractor(config)
    
    test_cases = [
        "Dave joined the meeting",
        "ML models need more training data",
        "The PM scheduled a review with the EM",
        "AWS services are used by the team"
    ]
    
    for i, test_text in enumerate(test_cases, 1):
        print(f"\n  Test {i}: '{test_text}'")
        entities = extractor.extract_entities(test_text)
        
        for entity in entities:
            original = entity['text']
            canonical = entity.get('canonical_name', original)
            rule = entity.get('disambiguation_rule', 'none')
            
            if original != canonical:
                print(f"    ✅ {original} → {canonical} (rule: {rule})")
            else:
                print(f"    ➖ {original} (no disambiguation)")


def test_performance():
    """Test extraction performance."""
    print("\n⚡ Testing Performance...")
    
    import time
    
    config = load_test_config()
    extractor = EnhancedEntityExtractor(config)
    
    # Test with longer text
    test_text = """
    Sprint Planning Meeting - July 19, 2025
    
    Attendees: Dave (Engineering Manager), Yolanda Li (Product Manager), 
    Glen Pink (Senior Software Engineer), Ben (Tech Lead)
    
    Agenda:
    1. Review Q3 roadmap and priorities
    2. Discuss AI platform architecture 
    3. Plan migration to Kubernetes
    4. Review CI/CD pipeline implementation
    
    Discussion:
    - Dave mentioned that the engineering team has been working on the React frontend
    - Yolanda shared updates from the product stakeholders at Google
    - Glen provided technical insights on the Django backend migration
    - Ben outlined the DevOps strategy for AWS deployment
    
    Action Items:
    - Schedule architecture review meeting with CTO
    - Create Jira tickets for the sprint backlog
    - Set up one-on-one meetings with each team member
    - Plan demo day for the AI features
    
    Next Meeting: Sprint Review on Friday
    """ * 3  # Make it 3x longer
    
    start_time = time.time()
    entities = extractor.extract_entities(test_text)
    relationships = extractor.infer_relationships(entities, test_text, {})
    end_time = time.time()
    
    processing_time = end_time - start_time
    
    print(f"📊 Processed {len(test_text)} characters in {processing_time:.2f} seconds")
    print(f"📊 Extracted {len(entities)} entities and {len(relationships)} relationships")
    print(f"📊 Rate: {len(test_text)/processing_time:.0f} chars/second")
    
    # Show pipeline stats
    stats = extractor.get_extraction_stats()
    print(f"📊 Pipeline components: {', '.join(stats['pipeline_components'])}")


def main():
    """Run all tests."""
    print("🚀 Enhanced Entity Extractor Test Suite")
    print("=" * 50)
    
    try:
        # Test basic extraction
        entities = test_basic_extraction()
        
        # Test relationships
        relationships = test_relationship_inference()
        
        # Test custom patterns
        custom_entities = test_custom_patterns()
        
        # Test disambiguation
        test_disambiguation()
        
        # Test performance
        test_performance()
        
        print("\n✅ All tests completed successfully!")
        print(f"📊 Summary: {len(entities)} entities, {len(relationships)} relationships")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 