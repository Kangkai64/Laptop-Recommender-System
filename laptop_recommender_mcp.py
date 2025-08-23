#!/usr/bin/env python3
"""
Laptop Recommender System MCP Server

This MCP server provides laptop recommendations based on user preferences
and specifications. It uses a collaborative filtering approach combined
with content-based filtering for personalized recommendations.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import mcp.server
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
    LoggingLevel,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LaptopRecommenderMCP:
    def __init__(self):
        self.data = None
        self.feature_matrix = None
        self.similarity_matrix = None
        self.scaler = StandardScaler()
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.load_data()
        self.preprocess_data()
    
    def load_data(self):
        """Load and clean the laptop dataset."""
        try:
            data_path = Path("data/Cleaned_Laptop_data.csv")
            self.data = pd.read_csv(data_path)
            logger.info(f"Loaded {len(self.data)} laptop records")
            
            # Clean the data
            self.data = self.data.dropna(subset=['latest_price', 'star_rating'])
            self.data['latest_price'] = pd.to_numeric(self.data['latest_price'], errors='coerce')
            self.data['star_rating'] = pd.to_numeric(self.data['star_rating'], errors='coerce')
            self.data = self.data.dropna(subset=['latest_price', 'star_rating'])
            
            # Create a combined features column for text-based similarity
            self.data['features'] = (
                self.data['brand'].astype(str) + ' ' +
                self.data['processor_brand'].astype(str) + ' ' +
                self.data['processor_name'].astype(str) + ' ' +
                self.data['ram_type'].astype(str) + ' ' +
                self.data['os'].astype(str)
            )
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self):
        """Preprocess data for recommendation algorithms."""
        try:
            # Create numerical features for content-based filtering
            numerical_features = [
                'ram_gb', 'ssd', 'hdd', 'graphic_card_gb', 
                'latest_price', 'star_rating', 'ratings'
            ]
            
            # Extract numeric values from string columns
            self.data['ram_gb_numeric'] = self.data['ram_gb'].str.extract(r'(\d+)').astype(float)
            self.data['ssd_numeric'] = self.data['ssd'].str.extract(r'(\d+)').astype(float)
            self.data['hdd_numeric'] = self.data['hdd'].str.extract(r'(\d+)').astype(float)
            # Handle graphic_card_gb which might be numeric already
            if self.data['graphic_card_gb'].dtype == 'object':
                self.data['graphic_card_gb_numeric'] = self.data['graphic_card_gb'].str.extract(r'(\d+)').astype(float)
            else:
                self.data['graphic_card_gb_numeric'] = self.data['graphic_card_gb'].astype(float)
            
            # Fill NaN values with 0
            self.data = self.data.fillna(0)
            
            # Create feature matrix
            feature_cols = [
                'ram_gb_numeric', 'ssd_numeric', 'hdd_numeric', 
                'graphic_card_gb_numeric', 'latest_price', 'star_rating', 'ratings'
            ]
            
            self.feature_matrix = self.data[feature_cols].values
            self.feature_matrix = self.scaler.fit_transform(self.feature_matrix)
            
            # Create text similarity matrix
            text_features = self.vectorizer.fit_transform(self.data['features'])
            self.similarity_matrix = cosine_similarity(text_features)
            
            logger.info("Data preprocessing completed")
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def get_recommendations_by_preferences(
        self, 
        max_price: Optional[float] = None,
        min_ram: Optional[int] = None,
        min_storage: Optional[int] = None,
        preferred_brands: Optional[List[str]] = None,
        min_rating: Optional[float] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Get laptop recommendations based on user preferences."""
        try:
            filtered_data = self.data.copy()
            
            # Apply filters
            if max_price:
                filtered_data = filtered_data[filtered_data['latest_price'] <= max_price]
            
            if min_ram:
                filtered_data = filtered_data[filtered_data['ram_gb_numeric'] >= min_ram]
            
            if min_storage:
                filtered_data = filtered_data[
                    (filtered_data['ssd_numeric'] >= min_storage) | 
                    (filtered_data['hdd_numeric'] >= min_storage)
                ]
            
            if preferred_brands:
                filtered_data = filtered_data[
                    filtered_data['brand'].str.lower().isin([b.lower() for b in preferred_brands])
                ]
            
            if min_rating:
                filtered_data = filtered_data[filtered_data['star_rating'] >= min_rating]
            
            if len(filtered_data) == 0:
                return []
            
            # Sort by rating and price (best value for money)
            filtered_data['value_score'] = (
                filtered_data['star_rating'] * filtered_data['ratings'] / 
                (filtered_data['latest_price'] / 1000)
            )
            
            recommendations = filtered_data.nlargest(top_k, 'value_score')
            
            return recommendations.to_dict('records')
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            return []
    
    def get_similar_laptops(self, laptop_id: int, top_k: int = 5) -> List[Dict[str, Any]]:
        """Get similar laptops based on content-based filtering."""
        try:
            if laptop_id >= len(self.data):
                return []
            
            # Get similarity scores for the given laptop
            similarity_scores = self.similarity_matrix[laptop_id]
            
            # Get indices of most similar laptops (excluding itself)
            similar_indices = similarity_scores.argsort()[::-1][1:top_k+1]
            
            similar_laptops = []
            for idx in similar_indices:
                laptop = self.data.iloc[idx].to_dict()
                laptop['similarity_score'] = float(similarity_scores[idx])
                similar_laptops.append(laptop)
            
            return similar_laptops
            
        except Exception as e:
            logger.error(f"Error getting similar laptops: {e}")
            return []
    
    def get_laptop_details(self, laptop_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific laptop."""
        try:
            if laptop_id >= len(self.data):
                return None
            
            return self.data.iloc[laptop_id].to_dict()
            
        except Exception as e:
            logger.error(f"Error getting laptop details: {e}")
            return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        try:
            stats = {
                'total_laptops': len(self.data),
                'brands': self.data['brand'].value_counts().to_dict(),
                'price_range': {
                    'min': float(self.data['latest_price'].min()),
                    'max': float(self.data['latest_price'].max()),
                    'mean': float(self.data['latest_price'].mean())
                },
                'rating_stats': {
                    'min': float(self.data['star_rating'].min()),
                    'max': float(self.data['star_rating'].max()),
                    'mean': float(self.data['star_rating'].mean())
                }
            }
            return stats
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}

# Global recommender instance
recommender = None

def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("laptop-recommender")
    
    @server.list_tools()
    async def handle_list_tools() -> ListToolsResult:
        """List available tools."""
        return ListToolsResult(
            tools=[
                Tool(
                    name="get_recommendations",
                    description="Get laptop recommendations based on user preferences",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "max_price": {"type": "number", "description": "Maximum price in INR"},
                            "min_ram": {"type": "integer", "description": "Minimum RAM in GB"},
                            "min_storage": {"type": "integer", "description": "Minimum storage in GB"},
                            "preferred_brands": {"type": "array", "items": {"type": "string"}, "description": "Preferred laptop brands"},
                            "min_rating": {"type": "number", "description": "Minimum star rating"},
                            "top_k": {"type": "integer", "description": "Number of recommendations to return", "default": 5}
                        }
                    }
                ),
                Tool(
                    name="get_similar_laptops",
                    description="Find laptops similar to a given laptop",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "laptop_id": {"type": "integer", "description": "ID of the reference laptop"},
                            "top_k": {"type": "integer", "description": "Number of similar laptops to return", "default": 5}
                        },
                        "required": ["laptop_id"]
                    }
                ),
                Tool(
                    name="get_laptop_details",
                    description="Get detailed information about a specific laptop",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "laptop_id": {"type": "integer", "description": "ID of the laptop"}
                        },
                        "required": ["laptop_id"]
                    }
                ),
                Tool(
                    name="get_statistics",
                    description="Get dataset statistics and insights",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="search_laptops",
                    description="Search laptops by brand, processor, or other criteria",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query (brand, processor, etc.)"},
                            "max_results": {"type": "integer", "description": "Maximum number of results", "default": 10}
                        },
                        "required": ["query"]
                    }
                )
            ]
        )
    
    @server.call_tool()
    async def handle_call_tool(name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Handle tool calls."""
        global recommender
        
        if recommender is None:
            recommender = LaptopRecommenderMCP()
        
        try:
            if name == "get_recommendations":
                recommendations = recommender.get_recommendations_by_preferences(
                    max_price=arguments.get("max_price"),
                    min_ram=arguments.get("min_ram"),
                    min_storage=arguments.get("min_storage"),
                    preferred_brands=arguments.get("preferred_brands"),
                    min_rating=arguments.get("min_rating"),
                    top_k=arguments.get("top_k", 5)
                )
                
                result_text = "## Laptop Recommendations\n\n"
                for i, laptop in enumerate(recommendations, 1):
                    result_text += f"### {i}. {laptop['brand']} {laptop['model']}\n"
                    result_text += f"- **Price:** ₹{laptop['latest_price']:,}\n"
                    result_text += f"- **Processor:** {laptop['processor_brand']} {laptop['processor_name']}\n"
                    result_text += f"- **RAM:** {laptop['ram_gb']}\n"
                    result_text += f"- **Storage:** SSD: {laptop['ssd']}, HDD: {laptop['hdd']}\n"
                    result_text += f"- **Rating:** {laptop['star_rating']} ⭐ ({laptop['ratings']} ratings)\n"
                    result_text += f"- **OS:** {laptop['os']} {laptop['os_bit']}\n\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result_text)]
                )
            
            elif name == "get_similar_laptops":
                laptop_id = arguments["laptop_id"]
                similar_laptops = recommender.get_similar_laptops(
                    laptop_id=laptop_id,
                    top_k=arguments.get("top_k", 5)
                )
                
                result_text = f"## Laptops Similar to ID {laptop_id}\n\n"
                for i, laptop in enumerate(similar_laptops, 1):
                    result_text += f"### {i}. {laptop['brand']} {laptop['model']}\n"
                    result_text += f"- **Similarity Score:** {laptop['similarity_score']:.3f}\n"
                    result_text += f"- **Price:** ₹{laptop['latest_price']:,}\n"
                    result_text += f"- **Processor:** {laptop['processor_brand']} {laptop['processor_name']}\n"
                    result_text += f"- **Rating:** {laptop['star_rating']} ⭐\n\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result_text)]
                )
            
            elif name == "get_laptop_details":
                laptop_id = arguments["laptop_id"]
                laptop = recommender.get_laptop_details(laptop_id)
                
                if laptop is None:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"Laptop with ID {laptop_id} not found.")]
                    )
                
                result_text = f"## Laptop Details (ID: {laptop_id})\n\n"
                result_text += f"**Brand:** {laptop['brand']}\n"
                result_text += f"**Model:** {laptop['model']}\n"
                result_text += f"**Price:** ₹{laptop['latest_price']:,}\n"
                result_text += f"**Processor:** {laptop['processor_brand']} {laptop['processor_name']} ({laptop['processor_generation']})\n"
                result_text += f"**RAM:** {laptop['ram_gb']} ({laptop['ram_type']})\n"
                result_text += f"**Storage:** SSD: {laptop['ssd']}, HDD: {laptop['hdd']}\n"
                result_text += f"**Graphics:** {laptop['graphic_card_gb']} GB\n"
                result_text += f"**OS:** {laptop['os']} {laptop['os_bit']}\n"
                result_text += f"**Display:** {laptop['display_size']}\"\n"
                result_text += f"**Weight:** {laptop['weight']}\n"
                result_text += f"**Touchscreen:** {'Yes' if laptop['touchscreen'] else 'No'}\n"
                result_text += f"**Microsoft Office:** {'Yes' if laptop['microsoft_office'] else 'No'}\n"
                result_text += f"**Rating:** {laptop['star_rating']} ⭐ ({laptop['ratings']} ratings, {laptop['reviews']} reviews)\n"
                result_text += f"**Warranty:** {laptop['warranty']} years\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result_text)]
                )
            
            elif name == "get_statistics":
                stats = recommender.get_statistics()
                
                result_text = "## Dataset Statistics\n\n"
                result_text += f"**Total Laptops:** {stats['total_laptops']}\n\n"
                
                result_text += "### Price Range (INR)\n"
                result_text += f"- **Minimum:** ₹{stats['price_range']['min']:,.0f}\n"
                result_text += f"- **Maximum:** ₹{stats['price_range']['max']:,.0f}\n"
                result_text += f"- **Average:** ₹{stats['price_range']['mean']:,.0f}\n\n"
                
                result_text += "### Rating Statistics\n"
                result_text += f"- **Minimum:** {stats['rating_stats']['min']:.1f} ⭐\n"
                result_text += f"- **Maximum:** {stats['rating_stats']['max']:.1f} ⭐\n"
                result_text += f"- **Average:** {stats['rating_stats']['mean']:.1f} ⭐\n\n"
                
                result_text += "### Top Brands\n"
                for brand, count in list(stats['brands'].items())[:10]:
                    result_text += f"- **{brand}:** {count} laptops\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result_text)]
                )
            
            elif name == "search_laptops":
                query = arguments["query"].lower()
                max_results = arguments.get("max_results", 10)
                
                # Search in brand, model, processor, and features
                search_mask = (
                    recommender.data['brand'].str.lower().str.contains(query) |
                    recommender.data['model'].str.lower().str.contains(query) |
                    recommender.data['processor_name'].str.lower().str.contains(query) |
                    recommender.data['features'].str.lower().str.contains(query)
                )
                
                search_results = recommender.data[search_mask].head(max_results)
                
                if len(search_results) == 0:
                    return CallToolResult(
                        content=[TextContent(type="text", text=f"No laptops found matching '{query}'.")]
                    )
                
                result_text = f"## Search Results for '{query}'\n\n"
                for i, laptop in enumerate(search_results.to_dict('records'), 1):
                    result_text += f"### {i}. {laptop['brand']} {laptop['model']}\n"
                    result_text += f"- **Price:** ₹{laptop['latest_price']:,}\n"
                    result_text += f"- **Processor:** {laptop['processor_brand']} {laptop['processor_name']}\n"
                    result_text += f"- **RAM:** {laptop['ram_gb']}\n"
                    result_text += f"- **Rating:** {laptop['star_rating']} ⭐\n\n"
                
                return CallToolResult(
                    content=[TextContent(type="text", text=result_text)]
                )
            
            else:
                return CallToolResult(
                    content=[TextContent(type="text", text=f"Unknown tool: {name}")]
                )
                
        except Exception as e:
            logger.error(f"Error in tool call {name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")]
            )
    
    return server

async def main():
    """Main entry point."""
    server = create_server()
    
    # Initialize the recommender system
    global recommender
    recommender = LaptopRecommenderMCP()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="laptop-recommender",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=None,
                    experimental_capabilities=None,
                ),
            ),
        )

if __name__ == "__main__":
    asyncio.run(main())
