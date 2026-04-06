"""
Graph Analyzer Implementation
Analyzes graph properties and identifies key nodes
"""
import asyncio
from typing import Dict, Any, List, Tuple
from datetime import datetime
import numpy as np

from .....utils.nlp_utils import nlp_utils
from .....config.settings import settings

try:
    import networkx as nx
    GRAPH_AVAILABLE = True
except ImportError:
    GRAPH_AVAILABLE = False


class GraphAnalyzer:
    """Graph analyzer implementation following detailed specification"""
    
    async def analyze_graph_properties(self, graph) -> Dict[str, Any]:
        """Analyze graph properties"""
        if not GRAPH_AVAILABLE or not graph:
            return await self._fallback_analysis()
        
        try:
            # Basic properties
            properties = {
                'node_count': graph.number_of_nodes(),
                'edge_count': graph.number_of_edges(),
                'density': nx.density(graph),
                'is_directed': graph.is_directed()
            }
            
            # Centrality measures
            centrality = await self._calculate_centrality_measures(graph)
            properties['centrality'] = centrality
            
            # Community detection
            communities = await self._detect_communities(graph)
            properties['communities'] = communities
            
            # Path analysis
            paths = await self._analyze_paths(graph)
            properties['paths'] = paths
            
            return {
                'metrics': properties,
                'analysis_metadata': {
                    'method': 'networkx_analysis',
                    'analyzed_at': datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            print(f"Graph analysis failed: {e}")
            return {'metrics': {}, 'error': str(e)}
    
    async def identify_key_nodes(self, graph) -> List[Dict[str, Any]]:
        """Identify key nodes in the graph"""
        if not GRAPH_AVAILABLE or not graph:
            return []
        
        try:
            key_nodes = []
            
            # Calculate centrality measures
            centrality = await self._calculate_centrality_measures(graph)
            
            # Get all nodes with their data
            nodes_data = dict(graph.nodes(data=True))
            
            # Identify nodes with high centrality
            for node_id, node_data in nodes_data.items():
                node_centrality = centrality.get(node_id, {})
                
                # Calculate key node score
                key_score = self._calculate_key_node_score(node_centrality)
                
                if key_score > settings.KEY_NODE_THRESHOLD:
                    key_nodes.append({
                        'node_id': node_id,
                        'node_data': node_data,
                        'centrality_measures': node_centrality,
                        'key_score': key_score,
                        'importance_rank': 'high'
                    })
            
            # Sort by key score
            key_nodes.sort(key=lambda x: x['key_score'], reverse=True)
            
            return key_nodes
            
        except Exception as e:
            print(f"Key node identification failed: {e}")
            return []
    
    async def _calculate_centrality_measures(self, graph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures"""
        if not GRAPH_AVAILABLE:
            return {}
        
        centrality = {}
        
        try:
            # Degree centrality
            centrality['degree'] = dict(nx.degree_centrality(graph))
            
            # Betweenness centrality
            centrality['betweenness'] = dict(nx.betweenness_centrality(graph))
            
            # Closeness centrality
            centrality['closeness'] = dict(nx.closeness_centrality(graph))
            
            # Eigenvector centrality
            try:
                centrality['eigenvector'] = dict(nx.eigenvector_centrality(graph))
            except:
                centrality['eigenvector'] = {}
            
            # PageRank
            centrality['pagerank'] = dict(nx.pagerank(graph))
            
        except Exception as e:
            print(f"Centrality calculation failed: {e}")
        
        return centrality
    
    async def _detect_communities(self, graph) -> List[Dict[str, Any]]:
        """Detect communities in the graph"""
        if not GRAPH_AVAILABLE:
            return []
        
        try:
            # Convert to undirected for community detection
            undirected_graph = graph.to_undirected()
            
            # Use Louvain method if available
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(undirected_graph)
                
                community_list = []
                for node_id, community_id in communities.items():
                    if community_id not in community_list:
                        community_list[community_id] = []
                    community_list[community_id].append(node_id)
                
                return [{
                    'method': 'louvain',
                    'communities': community_list,
                    'modularity': community_louvain.modularity(communities, undirected_graph)
                }]
                
            except ImportError:
                # Fallback to connected components
                communities = list(nx.connected_components(undirected_graph))
                return [{
                    'method': 'connected_components',
                    'communities': communities,
                    'total_communities': len(communities)
                }]
                
        except Exception as e:
            print(f"Community detection failed: {e}")
            return []
    
    async def _analyze_paths(self, graph) -> Dict[str, Any]:
        """Analyze paths in the graph"""
        if not GRAPH_AVAILABLE:
            return {}
        
        try:
            # Shortest paths
            path_analysis = {}
            
            # Average shortest path length
            if graph.is_connected():
                path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
                lengths = [length for lengths_dict in path_lengths.values() for length in lengths_dict.values() if length != float('inf')]
                path_analysis['avg_shortest_path_length'] = np.mean(lengths) if lengths else 0
                path_analysis['diameter'] = max(lengths) if lengths else 0
            
            # Bridges (edges whose removal disconnects the graph)
            bridges = list(nx.bridges(graph))
            path_analysis['bridges'] = bridges
            path_analysis['bridge_count'] = len(bridges)
            
            return path_analysis
            
        except Exception as e:
            print(f"Path analysis failed: {e}")
            return {}
    
    def _calculate_key_node_score(self, centrality_measures: Dict[str, float]) -> float:
        """Calculate key node score from centrality measures"""
        if not centrality_measures:
            return 0.0
        
        # Weighted combination of centrality measures
        score = 0.0
        
        # Degree centrality (weight: 0.3)
        degree = centrality_measures.get('degree', 0)
        score += degree * 0.3
        
        # Betweenness centrality (weight: 0.3)
        betweenness = centrality_measures.get('betweenness', 0)
        score += betweenness * 0.3
        
        # PageRank (weight: 0.2)
        pagerank = centrality_measures.get('pagerank', 0)
        score += pagerank * 0.2
        
        # Eigenvector centrality (weight: 0.2)
        eigenvector = centrality_measures.get('eigenvector', 0)
        score += eigenvector * 0.2
        
        return score
    
    async def _fallback_analysis(self) -> Dict[str, Any]:
        """Fallback analysis when NetworkX is not available"""
        return {
            'metrics': {
                'node_count': 0,
                'edge_count': 0,
                'density': 0.0,
                'is_directed': False,
                'centrality': {},
                'communities': [],
                'paths': {}
            },
            'analysis_metadata': {
                'method': 'fallback',
                'analyzed_at': datetime.now().isoformat(),
                'networkx_available': False
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for graph analyzer"""
        return {
            'status': 'working',
            'networkx_available': GRAPH_AVAILABLE,
            'analysis_methods': [
                'centrality_measures',
                'community_detection',
                'path_analysis',
                'key_node_identification'
            ]
        }
