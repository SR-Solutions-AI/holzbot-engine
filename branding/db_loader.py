from __future__ import annotations

from typing import Any

from common.supabase_client import get_supabase_client


def fetch_tenant_branding(tenant_slug: str) -> dict[str, Any]:
    """
    Load tenant branding config from `public.tenants.config`.

    Expected structure (all optional):
      tenants.config.pdf = {
        offer_prefix: "CHH",
        handler_name: "Florian Siemer",
        offer_title: "Angebot für Ihr Chiemgauer Massivholzhaus",
        company: { ... },
        assets: {
          identity_image: "chiemgauer.png",
          show_offer_logos: true,
          offer_logos_image: "offer_logos.png"
        }
      }
    """
    if not tenant_slug:
        return {}

    supabase = get_supabase_client()
    res = (
        supabase.table("tenants")
        .select("slug, config")
        .eq("slug", tenant_slug)
        .limit(1)
        .execute()
    )
    data = (res.data or [])
    if not data:
        return {}
    cfg = (data[0].get("config") or {}) if isinstance(data[0], dict) else {}
    pdf_cfg = cfg.get("pdf") if isinstance(cfg, dict) else None
    return pdf_cfg if isinstance(pdf_cfg, dict) else {}



