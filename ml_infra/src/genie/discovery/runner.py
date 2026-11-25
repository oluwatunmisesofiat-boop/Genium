"""Discovery execution runner."""
from typing import Dict, List
from genie.discovery.planner import plan_discovery_queries
from genie.discovery.trust import canonicalize_url, get_trust_tier, is_blocked_domain


class DiscoveryRunner:
    async def run_discovery(self, product_name: str | None = None, nafdac_number: str | None = None) -> Dict:
        queries = plan_discovery_queries(product_name=product_name, nafdac_number=nafdac_number)
        return {
            "queries_executed": [q.query_string for q in queries],
            "staged_results_count": 0,
            "status": "completed",
        }
