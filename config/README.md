# Enhanced Entity Configuration

This directory contains configuration files for the enhanced entity extraction system.

## Setup Instructions

1. **Copy the template file:**
   ```bash
   cp enhanced_entity_config.template.yaml enhanced_entity_config.yaml
   ```

2. **Customize your configuration:**
   - Edit `enhanced_entity_config.yaml` with your specific entities
   - Replace placeholder names with actual people in your organization
   - Add organization-specific abbreviations and terminology  
   - Customize product/project names relevant to your domain

3. **Keep your config private:**
   - The actual `enhanced_entity_config.yaml` file is automatically ignored by git
   - Only the template file is tracked in version control
   - This prevents sensitive names and internal terminology from being shared

## Configuration Sections

### `entity.disambiguation_rules`
Maps common abbreviations or variations to their full canonical forms:
```yaml
disambiguation_rules:
  "Dave": "David Smith"      # Person name normalization
  "ML": "Machine Learning"   # Technical abbreviation
  "PM": "Product Manager"    # Role abbreviation
```

### `entity.aliases`  
Maps different name variations to the same canonical entity:
```yaml
aliases:
  "David S": "David Smith"     # Partial name → full name
  "D. Smith": "David Smith"    # Initial → full name
  "AWS": "Amazon Web Services" # Common abbreviation
```

### `entity.domain_mappings`
Maps entity types to domain-specific canonical forms:
```yaml
domain_mappings:
  PERSON:
    "engineer": "Software Engineer"
  ORG:
    "team": "Team"
```

### `entity.custom_patterns`
Lists of terms to recognize as specific entity types:
```yaml
custom_patterns:
  MEETING_TYPE:
    - "standup"
    - "sprint planning"
  TECHNOLOGY:
    - "python"
    - "kubernetes"
```

## Usage

The enhanced entity extractor will automatically load your configuration and apply:
- **Disambiguation rules** to normalize entity names
- **Alias resolution** to consolidate variations  
- **Custom patterns** to detect domain-specific entities
- **Confidence scoring** based on your configured weights

## Security Notes

- Never commit your actual `enhanced_entity_config.yaml` file
- Only share the template file publicly
- Consider using environment variables for highly sensitive mappings
- Review the template before sharing to ensure no sensitive info leaks through 