import json
from typing import Dict, Any

class PromptManager:
    def __init__(self, prompts_file: str = "prompts.json"):
        self.prompts_file = prompts_file
        self._prompts = None
    
    @property
    def prompts(self) -> Dict[str, Any]:
        if self._prompts is None:
            self._load_prompts()
        return self._prompts
    
    def _load_prompts(self):
        """Load prompts from JSON file"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                self._prompts = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file {self.prompts_file} not found")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in prompts file: {e}")
    
    def get_prompt(self, category: str, template_key: str = "user_template", **kwargs) -> str:
        """Get formatted prompt"""
        try:
            template = self.prompts[category][template_key]
            return template.format(**kwargs)
        except KeyError as e:
            raise KeyError(f"Prompt not found: {category}.{template_key}")
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")
    
    def get_system_prompt(self, category: str) -> str:
        """Get system prompt for category"""
        return self.prompts[category].get("system", "")
    
    def reload(self):
        """Reload prompts from file"""
        self._prompts = None
        self._load_prompts()

# Global instance
prompt_manager = PromptManager()