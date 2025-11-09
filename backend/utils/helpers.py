"""
Helper utility functions
"""
import logging
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def fetch_huggingface_model_details(model_name: str) -> dict:
    """Fetch model details from Hugging Face"""
    try:
        # Try to find the model on Hugging Face
        search_url = f"https://huggingface.co/api/models?search={model_name}"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            models = response.json()
            if models and len(models) > 0:
                # Get the first matching model
                model_id = models[0].get('id', model_name)
                
                # Fetch detailed model info
                model_url = f"https://huggingface.co/{model_id}"
                model_response = requests.get(model_url, timeout=10)
                
                if model_response.status_code == 200:
                    soup = BeautifulSoup(model_response.text, 'html.parser')
                    
                    # Extract model card information
                    model_details = {
                        'model_id': model_id,
                        'model_name': model_name,
                        'found_on_hf': True,
                        'url': model_url
                    }
                    
                    # Try to extract parameters from the page
                    # Look for common patterns like "7B", "70B", "13B" parameters
                    text_content = soup.get_text()
                    
                    # Extract parameter count
                    import re
                    param_patterns = [
                        r'(\d+\.?\d*)\s*[Bb]illion',
                        r'(\d+\.?\d*)\s*B\s+parameters',
                        r'(\d+\.?\d*)\s*B\s+param',
                    ]
                    
                    for pattern in param_patterns:
                        match = re.search(pattern, text_content)
                        if match:
                            model_details['parameters'] = f"{match.group(1)}B"
                            break
                    
                    # Extract architecture info
                    if 'transformer' in text_content.lower():
                        model_details['architecture'] = 'Transformer'
                    elif 'diffusion' in text_content.lower():
                        model_details['architecture'] = 'Diffusion'
                    elif 'bert' in model_name.lower():
                        model_details['architecture'] = 'BERT'
                    elif 'gpt' in model_name.lower():
                        model_details['architecture'] = 'GPT'
                    elif 'llama' in model_name.lower():
                        model_details['architecture'] = 'LLaMA'
                    
                    # Extract model size/memory requirements from name
                    size_match = re.search(r'(\d+)[bB]', model_name)
                    if size_match:
                        model_details['size_billions'] = int(size_match.group(1))
                    
                    return model_details
        
        # If not found on HF, return basic info
        return {
            'model_name': model_name,
            'found_on_hf': False,
            'estimated_size': 'unknown'
        }
        
    except Exception as e:
        logger.error(f"Error fetching HuggingFace model details: {str(e)}")
        return {
            'model_name': model_name,
            'found_on_hf': False,
            'error': str(e)
        }
