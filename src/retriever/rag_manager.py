from .embeddings import EmbeddingManager
import json
import os
import numpy as np
from typing import List, Dict, Optional
from judge.coherence import judge_coherence
from judge.salience import judge_ad_salience
from judge.helpfulness import judge_helpfulness

class RAGManager:
    def __init__(self, index_path: Optional[str] = None):
        self.embedding_manager = EmbeddingManager()
        self.index_path = index_path or "data/rag_index"
        self._load_or_create_index()
        self.min_relevance_score = 0.6  # Minimum relevance score threshold

    def _load_or_create_index(self):
        """Load existing index or create a new one if it doesn't exist."""
        try:
            self.embedding_manager.load_index(self.index_path)
            print(f"✅ Loaded existing RAG index from {self.index_path}")
        except (ValueError, FileNotFoundError):
            print("⚠️ No existing index found. Will create new index when ads are added.")

    def add_ads_to_index(self, ads: List[Dict]):
        """Add ads to the RAG index."""
        # Process ads into text format for embedding
        texts = []
        metadata = []
        
        for ad in ads:
            # Create a rich text representation of the ad
            text = f"""Product: {ad['ad_product']}
Brand: {ad['brand']}
Description: {ad['description']}
Keywords: {', '.join(ad['keywords'])}
Target Audience: {ad['target_audience']}
Use Cases: {', '.join(ad['use_cases'])}"""
            
            # Store text and metadata
            texts.append(text)
            metadata.append({
                "ad_id": ad["ad_id"],
                "ad_data": ad,
                "success_rate": ad.get("success_rate", 0.5)
            })
        
        # Add to index
        self.embedding_manager.build_index(texts, metadata)
        
        # Save index
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        self.embedding_manager.save_index(self.index_path)
        
        print(f"✅ Added {len(ads)} ads to RAG index")

    def get_relevant_ads(self, query: str, k: int = 5) -> List[Dict]:
        """Get the k most relevant ads for a query."""
        # Get similar ads
        results = self.embedding_manager.search(query, k=k)
        
        # Enhance results with contextual scoring
        enhanced_results = []
        for result in results:
            # Get the ad text and data
            ad_text = result["text"]
            ad_data = result["metadata"]["ad_data"]
            
            # Calculate contextual relevance score
            contextual_score = self._calculate_contextual_score(query, ad_text, ad_data)
            
            # Calculate final score (combine embedding similarity with contextual score)
            final_score = 0.7 * result["score"] + 0.3 * contextual_score
            
            enhanced_results.append({
                "text": ad_text,
                "ad_data": ad_data,
                "score": result["score"],
                "contextual_score": contextual_score,
                "final_score": final_score
            })
        
        # Sort by final score
        enhanced_results.sort(key=lambda x: x["final_score"], reverse=True)
        return enhanced_results

    def _calculate_contextual_score(self, query: str, ad_text: str, ad_data: Dict) -> float:
        """Calculate a contextual relevance score between a query and an ad."""
        # Start with the success rate as a base score
        score = ad_data.get("success_rate", 0.5)
        
        # Check if query matches any successful queries
        if "successful_queries" in ad_data:
            for past_query in ad_data["successful_queries"]:
                if any(word in past_query.lower() for word in query.lower().split()):
                    score += 0.1  # Boost score for query term matches
        
        # Check keyword matches
        keywords = ad_data.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in query.lower():
                score += 0.1  # Boost score for keyword matches
        
        # Normalize score to [0, 1]
        return min(1.0, max(0.0, score))

    def evaluate_response(self, query: str, response: str, ad_text: str) -> Dict[str, float]:
        """Evaluate a generated response using the judge criteria."""
        scores = {
            "coherence": judge_coherence(response, query)["Coherence Score"],
            "salience": judge_ad_salience(query, response, ad_text)["Ad Salience Score"],
            "helpfulness": judge_helpfulness(response, query)["Helpfulness Score"]
        }
        scores["total"] = sum(scores.values())
        return scores

    def _create_rich_ad_text(self, ad: Dict) -> str:
        """Create a rich text representation of an ad with all available context."""
        sections = [
            f"Product: {ad.get('ad_product', '')}",
            f"Brand: {ad.get('brand', '')}",
            f"URL: {ad.get('url', '')}",
            f"Description: {ad.get('description', '')}",
            f"Keywords: {', '.join(ad.get('keywords', []))}",
            f"Target Audience: {ad.get('target_audience', '')}",
            f"Use Cases: {', '.join(ad.get('use_cases', []))}",
            f"Success Rate: {ad.get('success_rate', 0)}",
            f"Previous Queries: {', '.join(ad.get('successful_queries', []))}"
        ]
        return "\n".join(section for section in sections if not section.endswith(": "))

    def enhance_ad_context(self, query: str, ad_facts: Dict) -> Dict:
        """Enhance ad context with relevant information from similar successful ads."""
        # Get relevant ads similar to the current one
        ad_query = self._create_rich_ad_text(ad_facts)
        similar_ads = self.get_relevant_ads(ad_query, k=10)
        
        # Extract and merge useful information from similar ads
        enhanced_facts = ad_facts.copy()
        if similar_ads:
            keywords = set()
            use_cases = set()
            successful_queries = set()
            
            for similar in similar_ads:
                ad_data = similar["ad_data"]
                keywords.update(ad_data.get("keywords", []))
                use_cases.update(ad_data.get("use_cases", []))
                if "successful_queries" in ad_data:
                    successful_queries.update(ad_data["successful_queries"].split(", "))
            
            enhanced_facts.update({
                "keywords": list(keywords),
                "use_cases": list(use_cases),
                "similar_successful_queries": list(successful_queries),
                "similar_ads": [
                    {
                        "text": ad["text"],
                        "score": ad["final_score"],
                        "contextual_score": ad["contextual_score"]
                    }
                    for ad in similar_ads
                ]
            })
        
        return enhanced_facts

    def save_successful_ad(self, ad_facts: Dict, query: str, success_metrics: Dict):
        """Save successful ad interactions with enhanced metrics."""
        # Calculate success score based on judge metrics
        coherence_score = success_metrics.get("coherence_score", 0)
        salience_score = success_metrics.get("salience_score", 0)
        helpfulness_score = success_metrics.get("helpfulness_score", 0)
        
        # Normalize to 0-1 range
        success_rate = (coherence_score + salience_score + helpfulness_score) / 10
        
        ad_entry = {
            **ad_facts,
            "query_context": query,
            "success_metrics": success_metrics,
            "success_rate": success_rate,
            "timestamp": success_metrics.get("timestamp")
        }
        
        # Update successful queries list
        if "successful_queries" not in ad_facts:
            ad_facts["successful_queries"] = []
        ad_facts["successful_queries"].append(query)
        
        # Save to JSON file
        success_file = os.path.join(self.index_path, "successful_ads.json")
        existing_ads = []
        
        if os.path.exists(success_file):
            with open(success_file, "r") as f:
                existing_ads = json.load(f)
                
        existing_ads.append(ad_entry)
        
        with open(success_file, "w") as f:
            json.dump(existing_ads, f, indent=2)
            
        # Rebuild index with the new successful ad
        self.add_ads_to_index([ad_entry]) 