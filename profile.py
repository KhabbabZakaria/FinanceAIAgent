"""The user's investing context.

Edit this to match your own situation. It is the only input the critic agent has
about *you* -- without it, a recommendation is just arithmetic with no idea who
it is for.

Everything here is a constraint the critic checks the numbers against. Set a
value to None to switch that check off.
"""

USER_PROFILE = {
    # Free text, passed to the critic verbatim. Say what a human adviser
    # would need to know and the rules below cannot express.
    "summary": (
        "Long-horizon retail investor, mid-30s, investing monthly from salary. "
        "Comfortable with drawdowns in exchange for growth. No near-term need "
        "for this capital."
    ),

    "horizon_years": 20,
    "risk_tolerance": "high",          # low | medium | high

    # --- numeric constraints the critic checks deterministically ---
    "max_single_position": 0.35,       # no one holding above this share
    "max_sector_like_cluster": 0.60,   # combined cap on the tickers below
    "min_positions": 3,                # fewer than this is concentrated
    "target_fixed_income": 0.0,        # desired bond/cash share; 0 = not wanted
    "max_speculative": 0.15,           # combined cap on the speculative list

    # Buckets are declared, not inferred: guessing a sector from a ticker is
    # exactly the kind of recall the models get wrong.
    "speculative_tickers": {"BTC-USD", "ETH-USD", "PLTR", "RGTI", "QBTS",
                            "SERV", "MSTR", "COIN", "GME", "AMC"},
    "fixed_income_tickers": {"BND", "AGG", "TLT", "IEF", "SHY", "VGIT", "BNDX"},
    "cluster_tickers": {"AAPL", "MSFT", "NVDA", "GOOGL", "GOOG", "AMZN",
                        "META", "AVGO", "AMD", "INTC", "ORCL", "CRM"},
    "cluster_name": "big-tech",
}
