# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import sqlite3
import os
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import threading
import pickle
from dataclasses import dataclass
import math
import time
import re

# AI Model dependencies
HAS_CLIP = False
HAS_DINOV2 = False
HAS_FAISS = False
MODEL_ERROR = None

try:
    import torch
    import clip
    HAS_CLIP = True
    print("✅ CLIP model available")
except ImportError:
    MODEL_ERROR = "CLIP not available. Install with: pip install torch clip-by-openai"

try:
    import transformers
    from transformers import Dinov2Model, AutoImageProcessor
    HAS_DINOV2 = True
    print("✅ DINOv2 model available")
except ImportError:
    if not HAS_CLIP:
        MODEL_ERROR = "Neither CLIP nor DINOv2 available. Install with: pip install torch clip-by-openai transformers"

try:
    import faiss
    HAS_FAISS = True
    print("✅ FAISS vector search available")
except ImportError:
    MODEL_ERROR = "FAISS not available. Install with: pip install faiss-cpu"

# OCR fallback
HAS_OCR = False
try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    pass

@dataclass
class PricingData:
    """Pricing information for different card conditions from Pokepricetracker"""
    def __init__(self):
        # Per-provider comprehensive map: provider -> {'raw': {...}, 'graded': {...}, 'other': {...}, 'sales': {...}}
        self.providers = {}
        # Raw/Ungraded conditions
        self.nm = None  # Near Mint
        self.lp = None  # Lightly Played
        self.mp = None  # Moderately Played
        self.hp = None  # Heavily Played
        self.dmg = None  # Damaged
        
        # PSA Graded
        self.psa10 = None
        self.psa9 = None
        self.psa8 = None
        self.psa7 = None
        self.psa6 = None
        self.psa5 = None
        
        # BGS Graded
        self.bgs10 = None
        self.bgs95 = None
        self.bgs9 = None
        self.bgs85 = None
        self.bgs8 = None
        
        # CGC Graded
        self.cgc10 = None
        self.cgc95 = None
        self.cgc9 = None
        
        self.market_price = None
        self.last_updated = None
        self.currency = "USD"
        self.source = "Pokepricetracker"
        self.provider = None  # eBay, TCGPlayer, etc.
        self.sales_volume = {}  # Sales count per condition
    
    def to_dict(self):
        return {
            'nm': self.nm,
            'lp': self.lp,
            'mp': self.mp,
            'hp': self.hp,
            'dmg': self.dmg,
            'psa10': self.psa10,
            'psa9': self.psa9,
            'psa8': self.psa8,
            'psa7': self.psa7,
            'psa6': self.psa6,
            'psa5': self.psa5,
            'bgs10': self.bgs10,
            'bgs95': self.bgs95,
            'bgs9': self.bgs9,
            'bgs85': self.bgs85,
            'bgs8': self.bgs8,
            'cgc10': self.cgc10,
            'cgc95': self.cgc95,
            'cgc9': self.cgc9,
            'market_price': self.market_price,
            'last_updated': self.last_updated,
            'currency': self.currency,
            'source': self.source,
            'provider': self.provider,
            'sales_volume': self.sales_volume,
            'providers': self.providers
        }

@dataclass
class CardMatch:
    """Result from visual card matching"""
    card_id: str
    name: str
    set_name: str
    set_number: str
    rarity: str
    confidence: float
    similarity_score: float
    image_url: str
    database_id: Optional[int] = None
    additional_info: Optional[Dict] = None
    pricing: Optional[PricingData] = None

@dataclass
class CardData:
    """Enhanced data class for Pokemon card information"""
    def __init__(self):
        self.name = ""
        self.card_id = ""
        self.set_number = ""
        self.set_name = ""
        self.hp = ""
        self.rarity = ""
        self.card_type = ""
        self.pokemon_type = ""
        self.artist = ""
        self.year = ""
        self.abilities = []
        self.attacks = []
        self.weakness = ""
        self.resistance = ""
        self.retreat_cost = ""
        self.flavor_text = ""
        self.image_path = ""
        self.confidence = 0.0
        self.evolves_from = ""
        self.regulation_mark = ""
        self.special_features = []
        self.embedding = None  # Store the visual embedding
        self.pricing = None  # Store pricing data
        
    def to_dict(self):
        return {
            'name': self.name,
            'card_id': self.card_id,
            'set_number': self.set_number,
            'set_name': self.set_name,
            'hp': self.hp,
            'rarity': self.rarity,
            'card_type': self.card_type,
            'pokemon_type': self.pokemon_type,
            'artist': self.artist,
            'year': self.year,
            'abilities': self.abilities,
            'attacks': self.attacks,
            'weakness': self.weakness,
            'resistance': self.resistance,
            'retreat_cost': self.retreat_cost,
            'flavor_text': self.flavor_text,
            'image_path': self.image_path,
            'confidence': self.confidence,
            'evolves_from': self.evolves_from,
            'regulation_mark': self.regulation_mark,
            'special_features': self.special_features,
            'pricing': self.pricing.to_dict() if self.pricing else None
        }

class PokepricetrackerAPI:
    """Pokepricetracker API integration for card data and pricing"""
    
    def __init__(self, api_key: str = "pokeprice_pro_4fbb6d9b7375090d2d8333973f752d0fa1a682696e7abf61"):
        self.api_key = api_key
        self.base_url = "https://www.pokemonpricetracker.com/api"
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Accept': 'application/json',
            'User-Agent': 'PokemonCardIdentifier/1.0'
        })
        self.rate_limit_delay = 0.2  # 200ms between requests
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)
        self.last_request_time = time.time()
    
    
    def get_prices_by_id(self, card_id: str) -> Optional[dict]:
        """Fetch comprehensive pricing by TCG card id using /prices?id={id}."""
        try:
            self._rate_limit()
            response = self.session.get(
                f"{self.base_url}/prices",
                params={"id": card_id},
                timeout=30,
            )
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    print(f"❌ Non-JSON response for pricing {card_id}")
                    return None
            elif response.status_code == 404:
                print(f"❌ Prices not found: {card_id}")
                return None
            elif response.status_code == 401:
                print("❌ Unauthorized: check API key for Pokepricetracker")
                return None
            else:
                print(f"❌ API error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Error getting pricing: {e}")
            return None

    def get_cards_from_set(self, set_id: str, limit: int = 250) -> List[Dict]:
        """Get cards from a specific set"""
        try:
            self._rate_limit()
            
            params = {
                'setId': set_id,
                'limit': limit
            }
            
            response = self.session.get(
                f"{self.base_url}/prices",
                params=params,
                timeout=30
            )
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    # Handle different response formats
                    if isinstance(data, list):
                        return data
                    elif isinstance(data, dict):
                        return data.get('data', data.get('cards', data.get('results', [])))
                    return []
                except ValueError:
                    print(f"❌ Non-JSON response for set {set_id}")
                    return []
            else:
                print(f"❌ API error getting set cards: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"❌ Error getting cards from set: {e}")
            return []
    
    def get_popular_sets(self) -> List[Dict]:
        """Get list of popular Pokemon sets"""
        # Since Pokepricetracker doesn't have a sets endpoint, we'll use a predefined list
        return [
            {"code": "sv8", "name": "Surging Sparks"},
            {"code": "sv7", "name": "Stellar Crown"},
            {"code": "sv6", "name": "Paldean Fates"},
            {"code": "sv5", "name": "Paradox Rift"},
            {"code": "sv4", "name": "151"},
            {"code": "sv3", "name": "Obsidian Flames"},
            {"code": "sv2", "name": "Paldea Evolved"},
            {"code": "sv1", "name": "Scarlet & Violet Base Set"},
            {"code": "swsh12pt5", "name": "Crown Zenith"},
            {"code": "swsh11", "name": "Lost Origin"},
            {"code": "swsh10", "name": "Astral Radiance"},
            {"code": "swsh9", "name": "Brilliant Stars"},
            {"code": "swsh8", "name": "Fusion Strike"},
            {"code": "swsh7", "name": "Evolving Skies"},
            {"code": "cel25", "name": "Celebrations"},
            {"code": "base1", "name": "Base Set"},
            {"code": "base2", "name": "Jungle"},
            {"code": "base3", "name": "Fossil"},
        ]
    
    
    def parse_pricing_data(self, card_data: Dict) -> Optional[PricingData]:
        """Parse Pokepricetracker pricing (from /prices) into a comprehensive structure.
        Populates per-provider maps and summary NM/LP/graded fields.
        """
        pricing = PricingData()
        try:
            if not isinstance(card_data, dict):
                return None

            # Some APIs may wrap results; unwrap common fields
            payload = card_data.get("data") if isinstance(card_data.get("data"), dict) else card_data

            providers = [
                "ebay", "tcgplayer", "cardmarket", "whatnot", "mercari", "stockx",
                "market", "prices"
            ]

            def coerce(val):
                if isinstance(val, (int, float)):
                    return float(val)
                if isinstance(val, dict):
                    for k in ("average", "avg", "median", "price", "value"):
                        if k in val and isinstance(val[k], (int, float)):
                            return float(val[k])
                return None

            def collect_conditions(d):
                out = {}
                if not isinstance(d, dict):
                    return out
                for k, v in d.items():
                    val = coerce(v)
                    if val is not None:
                        out[str(k)] = val
                return out

            # Build providers map
            for prov in list(payload.keys() if isinstance(payload, dict) else []):
                if prov not in providers:
                    # keep unknown blocks too (some sources vary)
                    providers.append(prov)

            for prov in providers:
                block = payload.get(prov)
                if not isinstance(block, dict):
                    continue

                raw = {}
                graded = {}
                other = {}
                sales = {}

                # Nested prices
                if isinstance(block.get("prices"), dict):
                    conds = collect_conditions(block["prices"])
                    for k, v in conds.items():
                        u = k.upper()
                        if u in {"NM", "LP", "MP", "HP", "DMG"}:
                            raw[k] = v
                        elif u.startswith(("PSA", "BGS", "CGC")):
                            graded[k] = v
                        else:
                            other[k] = v

                # Flattened keys (sometimes providers put conditions at top-level)
                flat = {k: v for k, v in block.items() if k not in ("prices", "sales")}
                conds = collect_conditions(flat)
                for k, v in conds.items():
                    u = k.upper()
                    if u in {"NM", "LP", "MP", "HP", "DMG"}:
                        raw.setdefault(k, v)
                    elif u.startswith(("PSA", "BGS", "CGC")):
                        graded.setdefault(k, v)
                    else:
                        other.setdefault(k, v)

                # Sales counts
                if isinstance(block.get("sales"), dict):
                    for k, v in block["sales"].items():
                        if isinstance(v, (int, float)):
                            sales[k] = v

                pricing.providers[prov] = {}
                if raw:    pricing.providers[prov]["raw"] = raw
                if graded: pricing.providers[prov]["graded"] = graded
                if other:  pricing.providers[prov]["other"] = other
                if sales:  pricing.providers[prov]["sales"] = sales

            # Populate summary fields for convenience (prefer eBay; else first provider with values)
            primary = None
            if "ebay" in pricing.providers and pricing.providers["ebay"].get("raw"):
                primary = pricing.providers["ebay"]
                pricing.provider = "eBay"
            else:
                for p, blk in pricing.providers.items():
                    if blk.get("raw") or blk.get("graded"):
                        primary = blk
                        pricing.provider = p
                        break

            if primary:
                raw = primary.get("raw", {})
                graded = primary.get("graded", {})
                pricing.nm  = raw.get("NM")
                pricing.lp  = raw.get("LP")
                pricing.mp  = raw.get("MP")
                pricing.hp  = raw.get("HP")
                pricing.dmg = raw.get("DMG")
                pricing.psa10 = graded.get("PSA10")
                pricing.psa9  = graded.get("PSA9")
                pricing.psa8  = graded.get("PSA8")
                pricing.psa7  = graded.get("PSA7")
                pricing.psa6  = graded.get("PSA6")
                pricing.psa5  = graded.get("PSA5")

                # Derive market price heuristic
                pricing.market_price = pricing.nm or pricing.lp or pricing.mp or pricing.hp

                # Sales volume (if available)
                if primary.get("sales"):
                    pricing.sales_volume = primary["sales"]

            pricing.last_updated = __import__("datetime").datetime.now().isoformat()
            return pricing if pricing.providers else None

        except Exception as e:
            print(f"❌ Error parsing pricing data: {e}")
            return None

    
    def _extract_price(self, price_data) -> Optional[float]:
        """Extract price from various formats"""
        if price_data is None:
            return None
        
        if isinstance(price_data, (int, float)):
            return float(price_data)
        
        if isinstance(price_data, dict):
            # Try common keys
            for key in ['average', 'avg', 'median', 'price', 'value']:
                if key in price_data and price_data[key] is not None:
                    return float(price_data[key])
        
        return None

class AdvancedImageProcessor:
    """Enhanced image preprocessing for better embeddings"""
    
    def __init__(self):
        self.target_size = (224, 224)  # Standard size for vision models
        
    def preprocess_for_embedding(self, image_path: str) -> Image.Image:
        """Preprocess image specifically for embedding generation"""
        # Load image
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Apply basic enhancements
        image = self.enhance_image(image)
        
        # Crop to card area if possible
        image = self.crop_to_card(image)
        
        # Resize to model input size
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        return image
    
    def enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply enhancements to improve visual features"""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Slight color enhancement
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def crop_to_card(self, image: Image.Image) -> Image.Image:
        """Try to detect and crop to just the card area"""
        # Convert to numpy for OpenCV processing
        img_array = np.array(image)
        
        # Try to detect card boundaries
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to find card edges
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest rectangular contour
        largest_area = 0
        best_contour = None
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > largest_area and area > 10000:  # Minimum area threshold
                # Check if it's roughly rectangular
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # Roughly rectangular
                    largest_area = area
                    best_contour = contour
        
        # If we found a good contour, crop to it
        if best_contour is not None:
            x, y, w, h = cv2.boundingRect(best_contour)
            
            # Add some padding
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.width - x, w + 2 * padding)
            h = min(image.height - y, h + 2 * padding)
            
            # Crop the image
            cropped = image.crop((x, y, x + w, y + h))
            return cropped
        
        return image  # Return original if no good crop found

class CLIPCardEmbedder:
    """CLIP-based card embedding system"""
    
    def __init__(self, model_name: str = "ViT-B/32"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self.processor = AdvancedImageProcessor()
        
        print(f"🔄 Loading CLIP model {model_name} on {self.device}...")
        self.load_model()
    
    def load_model(self):
        """Load CLIP model"""
        try:
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self.model.eval()
            print(f"✅ CLIP model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load CLIP model: {e}")
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image"""
        try:
            # Preprocess image
            image = self.processor.preprocess_for_embedding(image_path)
            
            # Convert to tensor
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                features = self.model.encode_image(image_tensor)
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None
    
    def generate_text_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text (useful for search)"""
        try:
            text_tokens = clip.tokenize([text]).to(self.device)
            
            with torch.no_grad():
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error generating text embedding: {e}")
            return None

class DINOv2CardEmbedder:
    """DINOv2-based card embedding system"""
    
    def __init__(self, model_name: str = "facebook/dinov2-base"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = None
        self.processor_hf = None
        self.processor = AdvancedImageProcessor()
        
        print(f"🔄 Loading DINOv2 model {model_name} on {self.device}...")
        self.load_model()
    
    def load_model(self):
        """Load DINOv2 model"""
        try:
            self.model = Dinov2Model.from_pretrained(self.model_name).to(self.device)
            self.processor_hf = AutoImageProcessor.from_pretrained(self.model_name)
            self.model.eval()
            print(f"✅ DINOv2 model loaded successfully")
        except Exception as e:
            raise Exception(f"Failed to load DINOv2 model: {e}")
    
    def generate_embedding(self, image_path: str) -> np.ndarray:
        """Generate embedding for a single image"""
        try:
            # Preprocess image
            image = self.processor.preprocess_for_embedding(image_path)
            
            # Process with HuggingFace processor
            inputs = self.processor_hf(images=image, return_tensors="pt").to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use the [CLS] token embedding
                features = outputs.last_hidden_state[:, 0]  # Shape: [1, hidden_size]
                features = features / features.norm(dim=-1, keepdim=True)  # Normalize
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

class CardEmbeddingDatabase:
    """Database for storing and searching card embeddings"""
    
    def __init__(self, db_path: str = "pokemon_cards_visual.db", index_path: str = "card_embeddings.faiss"):
        self.db_path = db_path
        self.index_path = index_path
        self.embedding_dim = None
        self.index = None
        self.card_mapping = []  # Maps FAISS index to card info
        
        # Initialize database
        self.init_database()
        
        # Load existing index if available
        self.load_index()
    
    def init_database(self):
        """Initialize SQLite database for card metadata with proper schema migration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS card_embeddings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                card_id TEXT UNIQUE,
                name TEXT,
                set_name TEXT,
                set_number TEXT,
                rarity TEXT,
                card_type TEXT,
                pokemon_type TEXT,
                hp TEXT,
                image_url TEXT,
                image_path TEXT,
                embedding_vector BLOB,
                embedding_type TEXT,
                api_data TEXT,
                pricing_data TEXT,
                pricing_updated TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_card_id ON card_embeddings(card_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_name ON card_embeddings(name)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_set ON card_embeddings(set_name)')
        
        conn.commit()
        conn.close()
        print("✅ Database schema initialized successfully")
    
    def add_card_embedding(self, card_data: Dict, embedding: np.ndarray, embedding_type: str) -> bool:
        """Add a card embedding to the database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Serialize embedding
            embedding_blob = pickle.dumps(embedding)
            
            # Extract card info
            card_id = card_data.get('id', card_data.get('card_id', ''))
            name = card_data.get('name', card_data.get('cardName', ''))
            
            # Handle set information
            set_name = ''
            if isinstance(card_data.get('set'), dict):
                set_name = card_data['set'].get('name', '')
            elif isinstance(card_data.get('set'), str):
                set_name = card_data.get('set', '')
            
            number = card_data.get('number', card_data.get('cardNumber', ''))
            
            # Extract number from ID if needed
            if not number and card_id and '-' in card_id:
                parts = card_id.split('-')
                if len(parts) > 1:
                    number = parts[-1]
            
            cursor.execute('''
                INSERT OR REPLACE INTO card_embeddings 
                (card_id, name, set_name, set_number, rarity, card_type, pokemon_type, hp, 
                 image_url, image_path, embedding_vector, embedding_type, api_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                card_id,
                name,
                set_name,
                number,
                card_data.get('rarity', ''),
                card_data.get('supertype', ''),
                ', '.join(card_data.get('types', [])) if card_data.get('types') else '',
                card_data.get('hp', ''),
                self._extract_image_url(card_data),
                card_data.get('local_image_path', ''),
                embedding_blob,
                embedding_type,
                json.dumps(card_data)
            ))
            
            conn.commit()
            conn.close()
            
            # Add to FAISS index
            self.add_to_faiss_index(embedding, card_id)
            
            return True
            
        except Exception as e:
            print(f"Error adding card embedding: {e}")
            return False
    
    def _extract_image_url(self, card_data: Dict) -> str:
        """Extract image URL from various formats"""
        # Check for nested images object
        if card_data.get('images'):
            images = card_data['images']
            if isinstance(images, dict):
                return images.get('large', images.get('small', ''))
            elif isinstance(images, str):
                return images
        
        # Check other common fields
        for field in ['imageUrl', 'image_url', 'image', 'img']:
            if card_data.get(field):
                return card_data[field]
        
        return ''
    
    def add_to_faiss_index(self, embedding: np.ndarray, card_id: str):
        """Add embedding to FAISS index"""
        if not HAS_FAISS:
            return
        
        try:
            # Initialize index if needed
            if self.index is None:
                self.embedding_dim = len(embedding)
                self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
            
            # Add embedding
            embedding = embedding.reshape(1, -1).astype('float32')
            self.index.add(embedding)
            
            # Update mapping
            self.card_mapping.append(card_id)
            
        except Exception as e:
            print(f"Error adding to FAISS index: {e}")
    
    def search_similar_cards(self, query_embedding: np.ndarray, top_k: int = 10) -> List[CardMatch]:
        """Search for similar cards using FAISS"""
        if not HAS_FAISS or self.index is None:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1).astype('float32')
            query_norm = np.linalg.norm(query_embedding)
            if query_norm > 0:
                query_embedding = query_embedding / query_norm
            
            # Search
            scores, indices = self.index.search(query_embedding, min(top_k, len(self.card_mapping)))
            
            # Get card details
            matches = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.card_mapping):
                    card_id = self.card_mapping[idx]
                    
                    cursor.execute('''
                        SELECT id, card_id, name, set_name, set_number, rarity, 
                               image_url, api_data, pricing_data
                        FROM card_embeddings WHERE card_id = ?
                    ''', (card_id,))
                    
                    result = cursor.fetchone()
                    if result:
                        api_data = json.loads(result[7]) if result[7] else {}
                        pricing_data = json.loads(result[8]) if result[8] else None
                        
                        # Create pricing object if data exists
                        pricing = None
                        if pricing_data:
                            pricing = PricingData()
                            for key, value in pricing_data.items():
                                if hasattr(pricing, key):
                                    setattr(pricing, key, value)
                        
                        match = CardMatch(
                            card_id=result[1],
                            name=result[2],
                            set_name=result[3],
                            set_number=result[4],
                            rarity=result[5],
                            confidence=float(score),
                            similarity_score=float(score),
                            image_url=result[6],
                            database_id=result[0],
                            additional_info=api_data,
                            pricing=pricing
                        )
                        matches.append(match)
            
            conn.close()
            return matches
            
        except Exception as e:
            print(f"Error searching similar cards: {e}")
            return []
    
    def update_card_pricing(self, card_id: str, pricing: PricingData) -> bool:
        """Update pricing data for a card"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE card_embeddings 
                SET pricing_data = ?, pricing_updated = CURRENT_TIMESTAMP
                WHERE card_id = ?
            ''', (json.dumps(pricing.to_dict()), card_id))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error updating card pricing: {e}")
            return False
    
    def save_index(self):
        """Save FAISS index to disk"""
        if HAS_FAISS and self.index is not None:
            try:
                faiss.write_index(self.index, self.index_path)
                
                # Save mapping
                mapping_path = self.index_path.replace('.faiss', '_mapping.pkl')
                with open(mapping_path, 'wb') as f:
                    pickle.dump(self.card_mapping, f)
                
                print(f"✅ Index saved to {self.index_path}")
                
            except Exception as e:
                print(f"Error saving index: {e}")
    
    def load_index(self):
        """Load FAISS index from disk"""
        if not HAS_FAISS:
            return
        
        try:
            if os.path.exists(self.index_path):
                self.index = faiss.read_index(self.index_path)
                
                # Load mapping
                mapping_path = self.index_path.replace('.faiss', '_mapping.pkl')
                if os.path.exists(mapping_path):
                    with open(mapping_path, 'rb') as f:
                        self.card_mapping = pickle.load(f)
                
                print(f"✅ Index loaded from {self.index_path}")
                print(f"📊 {len(self.card_mapping)} cards in index")
                
        except Exception as e:
            print(f"Error loading index: {e}")
    
    def get_stats(self) -> Dict:
        """Get database statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM card_embeddings')
        total_cards = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(DISTINCT set_name) FROM card_embeddings')
        unique_sets = cursor.fetchone()[0]
        
        cursor.execute('SELECT embedding_type, COUNT(*) FROM card_embeddings GROUP BY embedding_type')
        embedding_types = dict(cursor.fetchall())
        
        cursor.execute('SELECT COUNT(*) FROM card_embeddings WHERE pricing_data IS NOT NULL')
        cards_with_pricing = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_cards': total_cards,
            'unique_sets': unique_sets,
            'embedding_types': embedding_types,
            'faiss_index_size': len(self.card_mapping) if self.card_mapping else 0,
            'cards_with_pricing': cards_with_pricing
        }

class VisualCardIdentifier:
    """Main visual card identification system"""
    
    def __init__(self, model_type: str = "clip"):
        self.model_type = model_type
        self.embedder = None
        self.database = CardEmbeddingDatabase()
        self.api = PokepricetrackerAPI()
        
        # Initialize the appropriate model
        self.init_embedder()
    
    def init_embedder(self):
        """Initialize the embedding model"""
        try:
            if self.model_type == "clip" and HAS_CLIP:
                self.embedder = CLIPCardEmbedder()
            elif self.model_type == "dinov2" and HAS_DINOV2:
                self.embedder = DINOv2CardEmbedder()
            else:
                available_models = []
                if HAS_CLIP:
                    available_models.append("CLIP")
                if HAS_DINOV2:
                    available_models.append("DINOv2")
                
                if available_models:
                    # Default to first available model
                    if HAS_CLIP:
                        self.model_type = "clip"
                        self.embedder = CLIPCardEmbedder()
                    elif HAS_DINOV2:
                        self.model_type = "dinov2"
                        self.embedder = DINOv2CardEmbedder()
                else:
                    raise Exception("No embedding models available")
                    
        except Exception as e:
            raise Exception(f"Failed to initialize embedder: {e}")
    
    def identify_card(self, image_path: str, top_k: int = 5, fetch_pricing: bool = True) -> List[CardMatch]:
        """Identify card from image using visual similarity"""
        try:
            # Generate embedding for query image
            query_embedding = self.embedder.generate_embedding(image_path)
            if query_embedding is None:
                return []
            
            # Search for similar cards
            matches = self.database.search_similar_cards(query_embedding, top_k)
            
            # Fetch pricing from Pokepricetracker if requested
            if fetch_pricing:
                for match in matches:
                    if match.pricing is None:  # Only fetch if not already cached
                        print(f"🔍 Getting Pokepricetracker pricing for: {match.card_id}")
                        card_data = self.api.get_prices_by_id(match.card_id)
                        if card_data:
                            pricing = self.api.parse_pricing_data(card_data)
                            if pricing:
                                match.pricing = pricing
                                # Cache pricing in database
                                self.database.update_card_pricing(match.card_id, pricing)
                                print(f"✅ Cached pricing for {match.name}")
                            else:
                                print(f"❌ No pricing found for {match.name}")
            
            return matches
            
        except Exception as e:
            print(f"Error identifying card: {e}")
            return []
    
    def add_reference_card(self, card_data: Dict, image_path: str = None) -> bool:
        """Add a reference card to the database"""
        try:
            # Download image if needed
            if image_path is None:
                image_url = self._extract_image_url(card_data)
                if image_url:
                    image_path = self.download_card_image(image_url, card_data.get('id', 'unknown'))
            
            if not image_path or not os.path.exists(image_path):
                print(f"No image available for card {card_data.get('name', 'Unknown')}")
                return False
            
            # Generate embedding
            embedding = self.embedder.generate_embedding(image_path)
            if embedding is None:
                return False
            
            # Add to database
            card_data['local_image_path'] = image_path
            success = self.database.add_card_embedding(card_data, embedding, self.model_type)
            
            if success:
                print(f"✅ Added: {card_data.get('name', 'Unknown')} ({card_data.get('id', 'No ID')})")
            
            return success
            
        except Exception as e:
            print(f"Error adding reference card: {e}")
            return False
    
    def _extract_image_url(self, card_data: Dict) -> str:
        """Extract image URL from card data"""
        if card_data.get('images'):
            images = card_data['images']
            if isinstance(images, dict):
                return images.get('large', images.get('small', ''))
            elif isinstance(images, str):
                return images
        
        # Try other fields
        for field in ['imageUrl', 'image_url', 'image', 'img']:
            if card_data.get(field):
                return card_data[field]
        
        # Try to construct from ID pattern (Pokemon TCG standard)
        if card_data.get('id'):
            card_id = card_data['id']
            return f"https://images.pokemontcg.io/{card_id.replace('-', '/')}.png"
        
        return ''
    
    def download_card_image(self, image_url: str, card_id: str) -> Optional[str]:
        """Download card image from URL"""
        try:
            # Create images directory
            os.makedirs('card_images', exist_ok=True)
            
            # Generate filename
            filename = f"card_images/{card_id}.jpg"
            
            # Skip if already exists
            if os.path.exists(filename):
                return filename
            
            # Download image
            response = requests.get(image_url, timeout=30)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    f.write(response.content)
                return filename
                
        except Exception as e:
            print(f"Error downloading image: {e}")
        
        return None
    
    def build_reference_database(self, max_cards: int = 500, progress_callback=None) -> int:
        """Build reference database with cards from Pokepricetracker"""
        cards_added = 0
        
        try:
            # Get popular sets
            sets = self.api.get_popular_sets()
            
            for set_data in sets:
                if cards_added >= max_cards:
                    break
                
                set_id = set_data['code']
                set_name = set_data['name']
                
                if progress_callback:
                    progress_callback(f"Processing set: {set_name}")
                
                # Get cards from set
                cards = self.api.get_cards_from_set(set_id, limit=50)
                
                for card in cards:
                    if cards_added >= max_cards:
                        break
                    
                    # Ensure card has required fields
                    if not card.get('id') and not card.get('card_id'):
                        continue
                    
                    # Add set info if missing
                    if not card.get('set'):
                        card['set'] = {'name': set_name}
                    
                    if progress_callback:
                        card_name = card.get('name', card.get('cardName', 'Unknown'))
                        progress_callback(f"Adding: {card_name} ({cards_added+1}/{max_cards})")
                    
                    if self.add_reference_card(card):
                        cards_added += 1
                
                # Small delay between sets
                time.sleep(0.5)
            
            # Save the index
            self.database.save_index()
            
            return cards_added
            
        except Exception as e:
            print(f"Error building database: {e}")
            return cards_added

class VisualCardIdentifierGUI:
    """GUI for the visual card identification system"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎴 Visual Pokemon Card Identifier + Pokepricetracker")
        self.root.geometry("1700x1000")
        
        # Initialize system
        self.identifier = None
        self.current_image = None
        self.current_matches = []
        
        # Colors
        self.colors = {
            'primary': '#2563eb',
            'primary_dark': '#1d4ed8',
            'secondary': '#64748b',
            'background': '#f8fafc',
            'surface': '#ffffff',
            'accent': '#f59e0b',
            'success': '#10b981',
            'error': '#ef4444',
            'text_primary': '#1e293b',
            'text_secondary': '#64748b',
            'border': '#e2e8f0'
        }
        
        self.setup_styles()
        self.setup_ui()
        self.check_model_availability()
    
    def setup_styles(self):
        """Setup UI styles"""
        style = ttk.Style()
        
        # Modern button styles
        style.configure('Modern.TButton',
                       background=self.colors['primary'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 8))
        
        style.configure('Success.TButton',
                       background=self.colors['success'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 8))
        
        style.configure('Error.TButton',
                       background=self.colors['error'],
                       foreground='white',
                       borderwidth=0,
                       focuscolor='none',
                       padding=(15, 8))
        
        style.configure('Secondary.TButton',
                       background=self.colors['surface'],
                       foreground=self.colors['text_primary'],
                       borderwidth=1,
                       focuscolor='none',
                       padding=(12, 6))
    
    def setup_ui(self):
        """Setup the main UI"""
        self.root.configure(bg=self.colors['background'])
        
        # Header
        self.setup_header()
        
        # Main content
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Setup tabs
        self.setup_identification_tab()
        self.setup_database_tab()
        self.setup_setup_tab()
    
    def setup_header(self):
        """Setup header section"""
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg=self.colors['primary'])
        header_content.pack(expand=True, fill=tk.BOTH, padx=30, pady=25)
        
        # Title
        tk.Label(header_content,
                text="🎴 Visual Pokemon Card Identifier",
                font=('Segoe UI', 22, 'bold'),
                fg='white',
                bg=self.colors['primary']).pack(side=tk.LEFT)
        
        # Subtitle
        tk.Label(header_content,
                text="Powered by Pokepricetracker API",
                font=('Segoe UI', 12),
                fg='#bfdbfe',
                bg=self.colors['primary']).pack(side=tk.LEFT, padx=(20, 0))
        
        # Status indicator
        self.status_indicator = tk.Label(header_content,
                                        text="🔄 Initializing...",
                                        font=('Segoe UI', 10, 'bold'),
                                        fg='#fbbf24',
                                        bg=self.colors['primary'])
        self.status_indicator.pack(side=tk.RIGHT)
    
    def setup_identification_tab(self):
        """Setup card identification tab"""
        id_frame = tk.Frame(self.notebook, bg=self.colors['background'])
        self.notebook.add(id_frame, text="🔍 Identify Card")
        
        # Create paned window
        paned = tk.PanedWindow(id_frame, orient=tk.HORIZONTAL, bg=self.colors['background'])
        paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel - Image upload and preview
        left_panel = tk.Frame(paned, bg=self.colors['surface'], width=500)
        paned.add(left_panel, minsize=400)
        
        # Upload controls
        upload_frame = tk.Frame(left_panel, bg=self.colors['surface'])
        upload_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(upload_frame,
                text="📷 Upload Card Image",
                font=('Segoe UI', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['surface']).pack(anchor=tk.W, pady=(0, 10))
        
        upload_buttons = tk.Frame(upload_frame, bg=self.colors['surface'])
        upload_buttons.pack(fill=tk.X)
        
        ttk.Button(upload_buttons,
                  text="📁 Select Image",
                  style='Modern.TButton',
                  command=self.select_image).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(upload_buttons,
                  text="🔍 Identify + Price",
                  style='Success.TButton',
                  command=self.identify_current_image).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(upload_buttons,
                  text="🧹 Clear",
                  style='Secondary.TButton',
                  command=self.clear_identification).pack(side=tk.LEFT)
        
        # Image preview
        preview_frame = tk.Frame(left_panel, bg=self.colors['background'], relief='solid', bd=1)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 20))
        
        self.preview_label = tk.Label(preview_frame,
                                     text="🎴\n\nSelect a Pokemon card image\nfor visual identification + pricing\n\n"
                                          "✨ Works with any angle or lighting\n"
                                          "🔥 No OCR required\n"
                                          "🎯 Visual matching with Pokepricetracker\n"
                                          "💰 Real-time pricing data",
                                     font=('Segoe UI', 11),
                                     fg=self.colors['text_secondary'],
                                     bg=self.colors['background'],
                                     justify=tk.CENTER)
        self.preview_label.pack(expand=True)
        
        # Right panel - Results
        right_panel = tk.Frame(paned, bg=self.colors['surface'])
        paned.add(right_panel, minsize=700)
        
        self.setup_results_panel(right_panel)
    
    def setup_results_panel(self, parent):
        """Setup the results panel"""
        # Title
        results_title_frame = tk.Frame(parent, bg=self.colors['surface'])
        results_title_frame.pack(fill=tk.X, padx=20, pady=(20, 0))
        
        tk.Label(results_title_frame,
                text="🎯 Identification Results + Pricing",
                font=('Segoe UI', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.confidence_label = tk.Label(results_title_frame,
                                        text="",
                                        font=('Segoe UI', 10, 'bold'),
                                        fg=self.colors['text_secondary'],
                                        bg=self.colors['surface'])
        self.confidence_label.pack(side=tk.RIGHT)
        
        # Results list
        results_frame = tk.Frame(parent, bg=self.colors['surface'])
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Scrollable results
        canvas = tk.Canvas(results_frame, bg=self.colors['surface'])
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=canvas.yview)
        self.results_container = tk.Frame(canvas, bg=self.colors['surface'])
        
        self.results_container.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.results_container, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Initial empty state
        self.show_empty_results()
    
    def setup_database_tab(self):
        """Setup database management tab"""
        db_frame = tk.Frame(self.notebook, bg=self.colors['background'])
        self.notebook.add(db_frame, text="🗄️ Database")
        
        # Stats section
        stats_frame = tk.Frame(db_frame, bg=self.colors['surface'])
        stats_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(stats_frame,
                text="📊 Database Statistics",
                font=('Segoe UI', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['surface']).pack(anchor=tk.W, pady=(10, 15))
        
        self.stats_text = tk.Text(stats_frame,
                                 height=8,
                                 font=('Segoe UI', 10),
                                 bg=self.colors['background'],
                                 relief='flat',
                                 wrap=tk.WORD)
        self.stats_text.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        # Control buttons
        control_frame = tk.Frame(stats_frame, bg=self.colors['surface'])
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(control_frame,
                  text="🔄 Refresh Stats",
                  style='Secondary.TButton',
                  command=self.refresh_database_stats).pack(side=tk.LEFT, padx=(10, 10))
        
        ttk.Button(control_frame,
                  text="💾 Save Index",
                  style='Secondary.TButton',
                  command=self.save_database_index).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(control_frame,
                  text="🗑️ Clear Database",
                  style='Error.TButton',
                  command=self.clear_database).pack(side=tk.RIGHT, padx=(10, 10))
    
    def setup_setup_tab(self):
        """Setup database building tab"""
        setup_frame = tk.Frame(self.notebook, bg=self.colors['background'])
        self.notebook.add(setup_frame, text="⚙️ Setup")
        
        # Model selection
        model_frame = tk.Frame(setup_frame, bg=self.colors['surface'])
        model_frame.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(model_frame,
                text="🤖 AI Model Configuration",
                font=('Segoe UI', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['surface']).pack(anchor=tk.W, pady=(10, 15))
        
        # Model selection
        model_select_frame = tk.Frame(model_frame, bg=self.colors['surface'])
        model_select_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(model_select_frame,
                text="Model Type:",
                font=('Segoe UI', 10, 'bold'),
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.model_var = tk.StringVar(value="clip")
        
        clip_radio = tk.Radiobutton(model_select_frame,
                                   text="CLIP (Recommended)",
                                   variable=self.model_var,
                                   value="clip",
                                   bg=self.colors['surface'],
                                   font=('Segoe UI', 10))
        clip_radio.pack(side=tk.LEFT, padx=(20, 10))
        
        dinov2_radio = tk.Radiobutton(model_select_frame,
                                     text="DINOv2 (Alternative)",
                                     variable=self.model_var,
                                     value="dinov2",
                                     bg=self.colors['surface'],
                                     font=('Segoe UI', 10))
        dinov2_radio.pack(side=tk.LEFT, padx=(0, 10))
        
        # Database building
        build_frame = tk.Frame(setup_frame, bg=self.colors['surface'])
        build_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        tk.Label(build_frame,
                text="🏗️ Build Reference Database",
                font=('Segoe UI', 14, 'bold'),
                fg=self.colors['text_primary'],
                bg=self.colors['surface']).pack(anchor=tk.W, pady=(10, 15))
        
        # Options
        options_frame = tk.Frame(build_frame, bg=self.colors['surface'])
        options_frame.pack(fill=tk.X, padx=10)
        
        tk.Label(options_frame,
                text="Number of cards:",
                font=('Segoe UI', 10),
                bg=self.colors['surface']).pack(side=tk.LEFT)
        
        self.num_cards_var = tk.StringVar(value="500")
        num_cards_entry = tk.Entry(options_frame,
                                  textvariable=self.num_cards_var,
                                  width=10,
                                  font=('Segoe UI', 10))
        num_cards_entry.pack(side=tk.LEFT, padx=(10, 20))
        
        ttk.Button(options_frame,
                  text="🚀 Initialize System",
                  style='Modern.TButton',
                  command=self.initialize_system).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(options_frame,
                  text="🏗️ Build Database",
                  style='Success.TButton',
                  command=self.build_reference_database).pack(side=tk.LEFT)
        
        # Progress display
        self.progress_frame = tk.Frame(build_frame, bg=self.colors['surface'])
        self.progress_frame.pack(fill=tk.X, padx=10, pady=(20, 10))
        
        self.progress_text = tk.Text(self.progress_frame,
                                    height=10,
                                    font=('Segoe UI', 9),
                                    bg=self.colors['background'],
                                    relief='flat')
        self.progress_text.pack(fill=tk.X)
    
    def check_model_availability(self):
        """Check which models are available"""
        if not (HAS_CLIP or HAS_DINOV2) or not HAS_FAISS:
            self.show_installation_help()
        else:
            available_models = []
            if HAS_CLIP:
                available_models.append("CLIP")
            if HAS_DINOV2:
                available_models.append("DINOv2")
            
            self.status_indicator.configure(
                text=f"✅ Ready ({', '.join(available_models)} + FAISS + Pokepricetracker)",
                fg='#10b981'
            )
    
    def show_installation_help(self):
        """Show installation help dialog"""
        help_window = tk.Toplevel(self.root)
        help_window.title("🔧 AI Model Setup Required")
        help_window.geometry("700x550")
        help_window.transient(self.root)
        help_window.configure(bg=self.colors['background'])
        
        # Header
        header_frame = tk.Frame(help_window, bg=self.colors['error'], height=60)
        header_frame.pack(fill=tk.X)
        header_frame.pack_propagate(False)
        
        tk.Label(header_frame,
                text="🔧 AI Model Setup Required",
                font=('Segoe UI', 16, 'bold'),
                fg='white',
                bg=self.colors['error']).pack(expand=True)
        
        # Content
        content_frame = tk.Frame(help_window, bg=self.colors['surface'])
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Error info
        error_text = MODEL_ERROR or "AI models not available"
        tk.Label(content_frame,
                text=f"❌ Issue: {error_text}",
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['error'],
                bg=self.colors['surface'],
                wraplength=650,
                justify=tk.LEFT).pack(anchor=tk.W, pady=(0, 20))
        
        # Instructions
        instructions = """
🚀 Visual Card Identification Setup:

Requirements:
1. PyTorch (deep learning framework):
   pip install torch torchvision

2. At least ONE of these models:
   
   🎯 CLIP (Recommended):
   pip install clip-by-openai
   
   🔬 DINOv2 (Alternative):
   pip install transformers

3. FAISS (vector search):
   pip install faiss-cpu

4. Complete installation command:
   pip install torch clip-by-openai transformers faiss-cpu

🎯 What You'll Get:
   • Visual similarity matching
   • Works with any angle, lighting, or glare
   • Real-time pricing from Pokepricetracker
   • All conditions and graded prices
   • 95%+ accuracy with sufficient reference database

⚡ Quick Install:
Copy and run: pip install torch clip-by-openai transformers faiss-cpu

🔄 After installation, restart this application.
"""
        
        text_widget = scrolledtext.ScrolledText(content_frame,
                                               wrap=tk.WORD,
                                               height=18,
                                               font=('Segoe UI', 10),
                                               bg=self.colors['background'],
                                               relief='flat',
                                               padx=10,
                                               pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, instructions)
        text_widget.configure(state='disabled')
        
        # Buttons
        button_frame = tk.Frame(content_frame, bg=self.colors['surface'])
        button_frame.pack(fill=tk.X, pady=(20, 0))
        
        ttk.Button(button_frame,
                  text="🔄 Check Again",
                  style='Modern.TButton',
                  command=lambda: [help_window.destroy(), self.check_model_availability()]).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame,
                  text="📋 Copy Install Command",
                  style='Secondary.TButton',
                  command=lambda: self.copy_install_command()).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(button_frame,
                  text="Continue (Limited Features)",
                  style='Secondary.TButton',
                  command=help_window.destroy).pack(side=tk.RIGHT)
        
        # Update status
        self.status_indicator.configure(
            text="❌ Models not available",
            fg=self.colors['error']
        )
    
    def copy_install_command(self):
        """Copy installation command to clipboard"""
        command = "pip install torch clip-by-openai transformers faiss-cpu"
        self.root.clipboard_clear()
        self.root.clipboard_append(command)
        messagebox.showinfo("Copied", "Installation command copied to clipboard!")
    
    def initialize_system(self):
        """Initialize the visual identification system"""
        if not (HAS_CLIP or HAS_DINOV2) or not HAS_FAISS:
            messagebox.showerror("Error", "Required AI models not available. Please install dependencies first.")
            return
        
        try:
            model_type = self.model_var.get()
            self.identifier = VisualCardIdentifier(model_type)
            
            self.status_indicator.configure(
                text=f"✅ {model_type.upper()} + Pokepricetracker ready",
                fg=self.colors['success']
            )
            
            messagebox.showinfo("Success", 
                               f"🎉 System initialized!\n\n"
                               f"• Visual Model: {model_type.upper()}\n"
                               f"• API: Pokepricetracker\n"
                               f"• Ready for identification and pricing")
            
            # Refresh database stats
            self.refresh_database_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize system: {str(e)}")
            self.status_indicator.configure(
                text="❌ Initialization failed",
                fg=self.colors['error']
            )
    
    def select_image(self):
        """Select an image file for identification"""
        file_path = filedialog.askopenfilename(
            title="Select Pokemon Card Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path).convert('RGB')
                
                # Resize for display
                display_size = (400, 560)
                image.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                photo = ImageTk.PhotoImage(image)
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
                
                # Store image path
                self.current_image = file_path
                
                # Clear previous results
                self.show_empty_results()
                
                messagebox.showinfo("Success", 
                                   "✅ Image loaded!\n\n"
                                   "Click 'Identify + Price' to find matching cards\n"
                                   "and get current pricing from Pokepricetracker")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def identify_current_image(self):
        """Identify the currently loaded image"""
        if not self.current_image:
            messagebox.showwarning("No Image", "Please select an image first")
            return
        
        if not self.identifier:
            messagebox.showwarning("System Not Ready", "Please initialize the system first in the Setup tab")
            return
        
        # Show progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Visual Identification + Pricing")
        progress_window.geometry("480x200")
        progress_window.transient(self.root)
        progress_window.grab_set()
        progress_window.configure(bg=self.colors['background'])
        
        progress_frame = tk.Frame(progress_window, bg=self.colors['surface'])
        progress_frame.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        tk.Label(progress_frame,
                text="🔍 Analyzing image + fetching pricing...",
                font=('Segoe UI', 12),
                bg=self.colors['surface']).pack(pady=(0, 10))
        
        progress_bar = ttk.Progressbar(progress_frame, mode='indeterminate')
        progress_bar.pack(fill=tk.X, pady=(0, 10))
        progress_bar.start()
        
        status_label = tk.Label(progress_frame,
                               text="Generating visual embedding...",
                               font=('Segoe UI', 10),
                               fg=self.colors['text_secondary'],
                               bg=self.colors['surface'])
        status_label.pack()
        
        def identify_thread():
            try:
                # Update status
                self.root.after(0, lambda: status_label.configure(text="🤖 Generating visual embedding..."))
                
                # Identify card with pricing
                matches = self.identifier.identify_card(self.current_image, top_k=10, fetch_pricing=True)
                
                self.root.after(0, lambda: status_label.configure(text="💰 Fetching Pokepricetracker pricing..."))
                
                # Update UI in main thread
                self.root.after(0, lambda: self.display_identification_results(matches))
                self.root.after(0, lambda: progress_window.destroy())
                
                if matches:
                    best_match = matches[0]
                    pricing_info = ""
                    if best_match.pricing and best_match.pricing.nm:
                        pricing_info = f"\nNM Price: ${best_match.pricing.nm:.2f}"
                    
                    self.root.after(0, lambda: messagebox.showinfo("Identification Complete",
                        f"🎯 Identification complete!\n\n"
                        f"Best match: {best_match.name}\n"
                        f"Similarity: {best_match.similarity_score:.1%}\n"
                        f"Set: {best_match.set_name}\n"
                        f"Number: {best_match.set_number}{pricing_info}\n\n"
                        f"Found {len(matches)} similar cards"))
                else:
                    self.root.after(0, lambda: messagebox.showwarning("No Matches",
                        "❌ No similar cards found in database.\n\n"
                        "Try building a larger database in Setup tab"))
                
            except Exception as e:
                self.root.after(0, lambda: progress_window.destroy())
                self.root.after(0, lambda: messagebox.showerror("Error",
                    f"❌ Identification failed: {str(e)}"))
        
        threading.Thread(target=identify_thread, daemon=True).start()
    
    def display_identification_results(self, matches: List[CardMatch]):
        """Display identification results with pricing"""
        # Clear previous results
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        if not matches:
            self.show_empty_results()
            return
        
        # Update confidence label
        best_confidence = matches[0].similarity_score
        confidence_text = f"Best Match: {best_confidence:.1%} similarity"
        if best_confidence > 0.8:
            confidence_color = self.colors['success']
        elif best_confidence > 0.6:
            confidence_color = self.colors['accent']
        else:
            confidence_color = self.colors['error']
        
        self.confidence_label.configure(text=confidence_text, fg=confidence_color)
        
        # Display matches
        for i, match in enumerate(matches):
            self.create_match_widget_with_pricing(match, i)
        
        self.current_matches = matches
    
    def create_match_widget_with_pricing(self, match: CardMatch, index: int):
        """Create a widget for a single match with pricing information"""
        # Match container
        match_frame = tk.Frame(self.results_container, 
                              bg=self.colors['background'], 
                              relief='solid', 
                              bd=1)
        match_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Header with rank and confidence
        header_frame = tk.Frame(match_frame, bg=self.colors['background'])
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        # Rank badge
        rank_color = self.colors['success'] if index == 0 else self.colors['secondary']
        rank_label = tk.Label(header_frame,
                             text=f"#{index + 1}",
                             font=('Segoe UI', 10, 'bold'),
                             bg=rank_color,
                             fg='white',
                             width=3)
        rank_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Card name
        name_label = tk.Label(header_frame,
                             text=match.name,
                             font=('Segoe UI', 12, 'bold'),
                             fg=self.colors['text_primary'],
                             bg=self.colors['background'])
        name_label.pack(side=tk.LEFT)
        
        # Similarity score
        similarity_text = f"{match.similarity_score:.1%}"
        similarity_label = tk.Label(header_frame,
                                   text=similarity_text,
                                   font=('Segoe UI', 10, 'bold'),
                                   fg=rank_color,
                                   bg=self.colors['background'])
        similarity_label.pack(side=tk.RIGHT)
        
        # Details
        details_frame = tk.Frame(match_frame, bg=self.colors['background'])
        details_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        details_text = f"Set: {match.set_name} | Number: {match.set_number} | Rarity: {match.rarity}"
        tk.Label(details_frame,
                text=details_text,
                font=('Segoe UI', 9),
                fg=self.colors['text_secondary'],
                bg=self.colors['background']).pack(anchor=tk.W)
        
        # Pokepricetracker Pricing section
        if match.pricing:
            pricing_frame = tk.Frame(match_frame, bg=self.colors['surface'], relief='solid', bd=1)
            pricing_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

            pricing_header = tk.Frame(pricing_frame, bg=self.colors['surface'])
            pricing_header.pack(fill=tk.X, padx=10, pady=(5, 5))

            tk.Label(pricing_header,
                    text=f"💰 Pokepricetracker Pricing (all providers)",
                    font=('Segoe UI', 10, 'bold'),
                    fg=self.colors['text_primary'],
                    bg=self.colors['surface']).pack(side=tk.LEFT)

            # Iterate providers in preferred order
            pref = ["ebay", "tcgplayer", "cardmarket", "whatnot", "mercari", "stockx", "market", "prices"]
            providers = list(match.pricing.providers.keys())
            ordered = [p for p in pref if p in providers] + [p for p in providers if p not in pref]

            for prov in ordered:
                blk = match.pricing.providers.get(prov) or {}
                if not any([blk.get("raw"), blk.get("graded"), blk.get("other")]):
                    continue

                sep = tk.Frame(pricing_frame, bg=self.colors['surface'])
                sep.pack(fill=tk.X, padx=10, pady=(6, 0))
                tk.Label(sep,
                        text=prov.upper(),
                        font=('Segoe UI', 10, 'bold'),
                        fg=self.colors['text_primary'],
                        bg=self.colors['surface']).pack(anchor=tk.W)

                # Raw
                if blk.get("raw"):
                    raw_frame = tk.Frame(pricing_frame, bg=self.colors['surface'])
                    raw_frame.pack(fill=tk.X, padx=20, pady=(2, 0))
                    tk.Label(raw_frame, text="Ungraded:", font=('Segoe UI', 9, 'bold'),
                             fg=self.colors['text_secondary'], bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))
                    for k in ["NM", "LP", "MP", "HP", "DMG"]:
                        if k in blk["raw"] and blk["raw"][k] is not None:
                            tk.Label(raw_frame,
                                     text=f"{k}: ${blk['raw'][k]:.2f}",
                                     font=('Segoe UI', 9),
                                     fg=self.colors['text_primary'],
                                     bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))

                # Graded
                if blk.get("graded"):
                    grd_frame = tk.Frame(pricing_frame, bg=self.colors['surface'])
                    grd_frame.pack(fill=tk.X, padx=20, pady=(2, 0))
                    tk.Label(grd_frame, text="Graded:", font=('Segoe UI', 9, 'bold'),
                             fg=self.colors['text_secondary'], bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))

                    # Sort PSA/BGS/CGC then by grade desc
                    for prefix in ["PSA", "BGS", "CGC"]:
                        items = [(k, v) for k, v in blk["graded"].items() if k.upper().startswith(prefix)]
                        if not items:
                            continue
                        # sort by numeric portion desc when possible
                        def grade_key(kv):
                            k = kv[0]
                            num = re.sub(r"[^0-9.]", "", k)
                            try:
                                return (0, -float(num))
                            except:
                                return (1, k)
                        for k, v in sorted(items, key=grade_key):
                            tk.Label(grd_frame,
                                     text=f"{k}: ${v:.2f}",
                                     font=('Segoe UI', 9),
                                     fg=self.colors['text_primary'],
                                     bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))

                # Other misc
                if blk.get("other"):
                    oth_frame = tk.Frame(pricing_frame, bg=self.colors['surface'])
                    oth_frame.pack(fill=tk.X, padx=20, pady=(2, 0))
                    tk.Label(oth_frame, text="Other:", font=('Segoe UI', 9, 'bold'),
                             fg=self.colors['text_secondary'], bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))
                    for k, v in blk["other"].items():
                        tk.Label(oth_frame,
                                 text=f"{k}: ${v:.2f}",
                                 font=('Segoe UI', 9),
                                 fg=self.colors['text_primary'],
                                 bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))

                # Sales
                if blk.get("sales"):
                    sales_frame = tk.Frame(pricing_frame, bg=self.colors['surface'])
                    sales_frame.pack(fill=tk.X, padx=20, pady=(2, 2))
                    tk.Label(sales_frame, text="Sales:", font=('Segoe UI', 9, 'bold'),
                             fg=self.colors['text_secondary'], bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))
                    for k, v in blk["sales"].items():
                        tk.Label(sales_frame,
                                 text=f"{k}: {v}",
                                 font=('Segoe UI', 9),
                                 fg=self.colors['text_secondary'],
                                 bg=self.colors['surface']).pack(side=tk.LEFT, padx=(0, 10))

        else:
            # No pricing available
            no_pricing_frame = tk.Frame(match_frame, bg=self.colors['background'])
            no_pricing_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
            
            tk.Label(no_pricing_frame,
                    text="💰 Pricing not available for this card",
                    font=('Segoe UI', 9, 'italic'),
                    fg=self.colors['text_secondary'],
                    bg=self.colors['background']).pack(anchor=tk.W)
            no_pricing_frame = tk.Frame(match_frame, bg=self.colors['background'])
            no_pricing_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
            
            tk.Label(no_pricing_frame,
                    text="💰 Pricing not available for this card",
                    font=('Segoe UI', 9, 'italic'),
                    fg=self.colors['text_secondary'],
                    bg=self.colors['background']).pack(anchor=tk.W)
        
        # Action buttons
        action_frame = tk.Frame(match_frame, bg=self.colors['background'])
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        
        if index == 0:  # Best match gets special treatment
            ttk.Button(action_frame,
                      text="📋 Use This Card",
                      style='Success.TButton',
                      command=lambda m=match: self.use_match(m)).pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(action_frame,
                  text="💰 Refresh Price",
                  style='Secondary.TButton',
                  command=lambda m=match: self.refresh_match_pricing(m)).pack(side=tk.LEFT)
    
    def refresh_match_pricing(self, match: CardMatch):
        """Refresh pricing for a specific match"""
        if not self.identifier:
            messagebox.showwarning("System Not Ready", "Please initialize the system first")
            return
        
        try:
            print(f"🔄 Refreshing Pokepricetracker pricing for: {match.card_id}")
            
            # Get fresh pricing
            card_data = self.identifier.api.get_prices_by_id(match.card_id)
            if card_data:
                pricing = self.identifier.api.parse_pricing_data(card_data)
                if pricing:
                    match.pricing = pricing
                    # Update database cache
                    self.identifier.database.update_card_pricing(match.card_id, pricing)
                    # Refresh display
                    self.display_identification_results(self.current_matches)
                    messagebox.showinfo("Success", f"✅ Pricing updated for {match.name}")
                else:
                    messagebox.showwarning("No Pricing", f"❌ No pricing found for {match.name}")
            else:
                messagebox.showwarning("Error", f"❌ Could not fetch card data")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to refresh pricing: {str(e)}")
    
    def use_match(self, match: CardMatch):
        """Use the selected match"""
        pricing_text = ""
        if match.pricing:
            if match.pricing.nm:
                pricing_text = f"\nNM Price: ${match.pricing.nm:.2f}"
            if match.pricing.market_price:
                pricing_text += f"\nMarket Price: ${match.pricing.market_price:.2f}"
        
        messagebox.showinfo("Match Selected",
                           f"✅ Selected: {match.name}\n\n"
                           f"Card ID: {match.card_id}\n"
                           f"Set: {match.set_name}\n"
                           f"Number: {match.set_number}\n"
                           f"Similarity: {match.similarity_score:.1%}{pricing_text}")
    
    def show_empty_results(self):
        """Show empty results state"""
        for widget in self.results_container.winfo_children():
            widget.destroy()
        
        empty_label = tk.Label(self.results_container,
                              text="🎯💰\n\nIdentification results will appear here\n\n"
                                   "Select an image and click 'Identify + Price'",
                              font=('Segoe UI', 11),
                              fg=self.colors['text_secondary'],
                              bg=self.colors['surface'],
                              justify=tk.CENTER)
        empty_label.pack(expand=True, fill=tk.BOTH, padx=20, pady=20)
        
        self.confidence_label.configure(text="")
    
    def clear_identification(self):
        """Clear current identification"""
        self.current_image = None
        self.current_matches = []
        
        # Reset preview
        self.preview_label.configure(
            image="",
            text="🎴\n\nSelect a Pokemon card image\nfor visual identification + pricing\n\n"
                 "✨ Works with any angle or lighting\n"
                 "🔥 No OCR required\n"
                 "🎯 Visual matching with Pokepricetracker\n"
                 "💰 Real-time pricing data"
        )
        self.preview_label.image = None
        
        # Clear results
        self.show_empty_results()
    
    def build_reference_database(self):
        """Build reference database with cards from Pokepricetracker"""
        if not self.identifier:
            messagebox.showwarning("System Not Ready", "Please initialize the system first")
            return
        
        try:
            max_cards = int(self.num_cards_var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid number of cards")
            return
        
        if max_cards < 1 or max_cards > 10000:
            messagebox.showerror("Invalid Range", "Number of cards must be between 1 and 10,000")
            return
        
        # Confirm action
        if not messagebox.askyesno("Build Database",
                                  f"Build reference database with {max_cards} cards?\n\n"
                                  f"This will:\n"
                                  f"• Download cards from Pokepricetracker API\n"
                                  f"• Generate AI embeddings for visual identification\n"
                                  f"• Create searchable index\n"
                                  f"• Take several minutes\n\n"
                                  f"Continue?"):
            return
        
        # Clear progress display
        self.progress_text.delete(1.0, tk.END)
        
        def progress_callback(message):
            self.progress_text.insert(tk.END, f"{message}\n")
            self.progress_text.see(tk.END)
            self.root.update()
        
        def build_thread():
            try:
                progress_callback("🚀 Starting database build with Pokepricetracker...")
                progress_callback(f"Target: {max_cards} cards")
                progress_callback("")
                
                cards_added = self.identifier.build_reference_database(
                    max_cards=max_cards,
                    progress_callback=progress_callback
                )
                
                self.root.after(0, lambda: messagebox.showinfo("Build Complete",
                    f"🎉 Database build complete!\n\n"
                    f"Added: {cards_added} cards\n"
                    f"Visual Model: {self.identifier.model_type.upper()}\n"
                    f"Source: Pokepricetracker API\n\n"
                    f"You can now identify cards!"))
                
                # Refresh stats
                self.root.after(0, lambda: self.refresh_database_stats())
                
            except Exception as e:
                error_msg = f"❌ Database build failed: {str(e)}"
                progress_callback(error_msg)
                self.root.after(0, lambda: messagebox.showerror("Build Failed", error_msg))
        
        threading.Thread(target=build_thread, daemon=True).start()
    
    def refresh_database_stats(self):
        """Refresh database statistics display"""
        if not self.identifier:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, "❌ System not initialized\n\nPlease initialize the system first in the Setup tab.")
            return
        
        try:
            stats = self.identifier.database.get_stats()
            
            stats_info = f"""📊 Visual Card Database Statistics

🎴 Total Cards: {stats['total_cards']:,}
📦 Unique Sets: {stats['unique_sets']:,}
🔍 FAISS Index Size: {stats['faiss_index_size']:,}
💰 Cards with Cached Pricing: {stats['cards_with_pricing']:,}

🤖 Embedding Types:
"""
            
            for embedding_type, count in stats['embedding_types'].items():
                stats_info += f"   • {embedding_type.upper()}: {count:,} cards\n"
            
            pricing_coverage = (stats['cards_with_pricing'] / max(stats['total_cards'], 1)) * 100
            stats_info += f"\n💰 Cached Pricing Coverage: {pricing_coverage:.1f}%"
            
            if stats['total_cards'] == 0:
                stats_info += "\n\n⚠️ No cards in database yet. Build reference database to enable identification."
            elif stats['total_cards'] < 100:
                stats_info += f"\n\n💡 Small database ({stats['total_cards']} cards). Consider adding more cards for better accuracy."
            else:
                stats_info += f"\n\n✅ Good database size for accurate identification!"
            
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, stats_info)
            
        except Exception as e:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(tk.END, f"❌ Error loading stats: {str(e)}")
    
    def save_database_index(self):
        """Save the current FAISS index"""
        if not self.identifier:
            messagebox.showwarning("System Not Ready", "Please initialize the system first")
            return
        
        try:
            self.identifier.database.save_index()
            messagebox.showinfo("Success", "✅ Database index saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save index: {str(e)}")
    
    def clear_database(self):
        """Clear the entire database"""
        if not messagebox.askyesno("Clear Database",
                                  "⚠️ This will permanently delete ALL cards and cached pricing!\n\n"
                                  "This action cannot be undone.\n\n"
                                  "Are you sure?"):
            return
        
        try:
            # Clear SQLite database
            conn = sqlite3.connect(self.identifier.database.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM card_embeddings')
            conn.commit()
            conn.close()
            
            # Clear FAISS index
            self.identifier.database.index = None
            self.identifier.database.card_mapping = []
            
            # Remove index files
            if os.path.exists(self.identifier.database.index_path):
                os.remove(self.identifier.database.index_path)
            
            mapping_path = self.identifier.database.index_path.replace('.faiss', '_mapping.pkl')
            if os.path.exists(mapping_path):
                os.remove(mapping_path)
            
            messagebox.showinfo("Success", "✅ Database cleared successfully!")
            self.refresh_database_stats()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to clear database: {str(e)}")

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = VisualCardIdentifierGUI(root)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()