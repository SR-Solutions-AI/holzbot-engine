# tables.py - Toate funcțiile de creare tabele pentru PDF
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.units import mm

from .styles import get_styles, COLORS, BOLD_FONT, BASE_FONT
from .utils import format_money, format_area, format_length, safe_get

def P(text: str, style_name: str = "Cell") -> Paragraph:
    styles = get_styles()
    return Paragraph(str(text).replace("\n", "<br/>"), styles[style_name])

# ----------------------------
# Client info
# ----------------------------
def create_client_info_table(client_data: dict) -> Table:
    rows = [
        [P("Client", "CellBold"), P(client_data.get("nume", "—"), "Cell")],
        [P("Telefon", "CellBold"), P(client_data.get("telefon", "—"), "Cell")],
        [P("Email", "CellBold"), P(client_data.get("email", "—"), "Cell")],
        [P("Localitate", "CellBold"), P(client_data.get("localitate", "—"), "Cell")],
        [P("Proiect", "CellBold"), P(client_data.get("referinta", "—"), "Cell")],
    ]
    table = Table(rows, colWidths=[40*mm, 130*mm])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, COLORS["border"]),
        ("BACKGROUND", (0,0), (0,-1), COLORS["bg_header"]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]))
    return table

# ----------------------------
# Inputs info (sistem constructiv, etc)
# ----------------------------
def create_inputs_info_table(inputs: dict) -> Table:
    """Tabel cu toate inputurile și caracteristicile proiectului"""
    label_map = {
        "tipSistem": "Sistem constructiv",
        "gradPrefabricare": "Grad prefabricare",
        "tipFundatie": "Tip fundație",
        "tipAcoperis": "Tip acoperiș",
        "nivelOferta": "Nivel ofertă",
        "finisajInterior": "Finisaj interior",
        "fatada": "Finisaj exterior",
        "tamplarie": "Tâmplărie",
        "materialAcoperis": "Material acoperiș",
        "nivelEnergetic": "Nivel energetic",
        "incalzire": "Tip încălzire",
        "ventilatie": "Ventilație",
    }
    
    rows = [[P("Caracteristică", "CellBold"), P("Valoare", "CellBold")]]
    
    for key, label in label_map.items():
        if key in inputs:
            val = inputs[key]
            if isinstance(val, bool):
                val = "Da" if val else "Nu"
            rows.append([P(label, "Cell"), P(str(val), "Cell")])
    
    # Adaugă și alte chei necunoscute
    for k, v in inputs.items():
        if k not in label_map and v:
            rows.append([P(str(k), "CellSmall"), P(str(v), "CellSmall")])

    table = Table(rows, colWidths=[70*mm, 100*mm])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, COLORS["border"]),
        ("BACKGROUND", (0,0), (-1,0), COLORS["bg_header"]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ]))
    return table

# ----------------------------
# Plan summary
# ----------------------------
def create_plan_summary_table(plan_data: dict, plan_id: str) -> Table:
    floor_type = safe_get(plan_data, "floor_type", default="unknown")
    house_area = safe_get(plan_data, "house_area_m2", default=0.0)

    floor_names = {
        "ground_floor": "Parter (Ground Floor)",
        "top_floor": "Etaj (Top Floor)",
        "intermediate": "Etaj Intermediar",
        "unknown": "Necunoscut",
    }

    rows = [
        [P("Plan ID", "CellBold"), P(plan_id, "Cell")],
        [P("Tip Etaj", "CellBold"), P(floor_names.get(floor_type, floor_type), "Cell")],
        [P("Suprafață Totală", "CellBold"), P(format_area(house_area), "Cell")],
    ]
    table = Table(rows, colWidths=[50*mm, 120*mm])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, COLORS["border"]),
        ("BACKGROUND", (0,0), (0,-1), COLORS["bg_light"]),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING", (0,0), (-1,-1), 5),
    ]))
    return table

# ----------------------------
# Construction & Surfaces (summary style, ca în offer_pdf vechi)
# ----------------------------
def create_construction_surfaces_table(pricing_data: dict, plan_data: dict) -> Table:
    """Tabel similar cu 'Konstruktionen & Oberflächen' din offer_pdf.py vechi"""
    bd = pricing_data.get("breakdown", {})
    
    # Structuri pereți
    walls_items = bd.get("structure_walls", {}).get("items", [])
    int_area = 0.0
    ext_area = 0.0
    int_cost = 0.0
    ext_cost = 0.0
    
    for it in walls_items:
        name = it.get("name", "").lower()
        area = float(it.get("area_m2", 0.0))
        cost = float(it.get("total_cost", 0.0))
        if "interior" in name:
            int_area += area
            int_cost += cost
        elif "exterior" in name:
            ext_area += area
            ext_cost += cost
    
    # Planșee & Tavane
    fc_items = bd.get("floors_ceilings", {}).get("items", [])
    floor_area = 0.0
    floor_cost = 0.0
    ceiling_area = 0.0
    ceiling_cost = 0.0
    
    for it in fc_items:
        name = it.get("name", "").lower()
        area = float(it.get("area_m2", 0.0))
        cost = float(it.get("total_cost", 0.0))
        if "podea" in name or "planșeu" in name:
            floor_area += area
            floor_cost += cost
        if "tavan" in name:
            ceiling_area += area
            ceiling_cost += cost
    
    # Fundație
    found_items = bd.get("foundation", {}).get("items", [])
    found_area = 0.0
    found_cost = 0.0
    if found_items:
        found_area = float(found_items[0].get("area_m2", 0.0))
        found_cost = float(found_items[0].get("total_cost", 0.0))
    
    head = [P("Element","CellBold"), P("Suprafață","CellBold"), P("Preț/m²","CellBold"), P("Total","CellBold")]
    data = []
    
    if found_cost > 0:
        data.append([
            P("Fundație / Placă","Cell"),
            P(format_area(found_area)),
            P(format_money(found_cost/found_area if found_area else 0),"CellSmall"),
            P(format_money(found_cost),"CellBold")
        ])
    
    if floor_cost > 0:
        data.append([
            P("Structură Planșeu","Cell"),
            P(format_area(floor_area)),
            P(format_money(floor_cost/floor_area if floor_area else 0),"CellSmall"),
            P(format_money(floor_cost),"CellBold")
        ])
    
    if int_cost > 0:
        data.append([
            P("Pereți Interiori – Structură","Cell"),
            P(format_area(int_area)),
            P(format_money(int_cost/int_area if int_area else 0),"CellSmall"),
            P(format_money(int_cost),"CellBold")
        ])
    
    if ext_cost > 0:
        data.append([
            P("Pereți Exteriori – Structură","Cell"),
            P(format_area(ext_area)),
            P(format_money(ext_cost/ext_area if ext_area else 0),"CellSmall"),
            P(format_money(ext_cost),"CellBold")
        ])
    
    if ceiling_cost > 0:
        data.append([
            P("Structură Tavan","Cell"),
            P(format_area(ceiling_area)),
            P(format_money(ceiling_cost/ceiling_area if ceiling_area else 0),"CellSmall"),
            P(format_money(ceiling_cost),"CellBold")
        ])
    
    # Finisaje
    fin_items = bd.get("finishes", {}).get("items", [])
    for it in fin_items:
        name = it.get("name", "Finisaj")
        area = float(it.get("area_m2", 0.0))
        cost = float(it.get("total_cost", 0.0))
        if cost > 0:
            data.append([
                P(name,"Cell"),
                P(format_area(area)),
                P(format_money(cost/area if area else 0),"CellSmall"),
                P(format_money(cost),"CellBold")
            ])
    
    if not data:
        data = [[P("—"), P("—"), P("—"), P("—")]]
    
    tbl = Table([head] + data, colWidths=[65*mm, 35*mm, 30*mm, 35*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ]))
    return tbl

# ----------------------------
# Foundation details
# ----------------------------
def create_foundation_table(foundation_bd: dict) -> Table:
    items = foundation_bd.get("items", [])
    head = [P("Descriere","CellBold"), P("Suprafață","CellBold"), P("Preț/m²","CellBold"), P("Total","CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Fundație")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(name,"Cell"),
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = foundation_bd.get("total_cost", 0.0)
    data.append([P("TOTAL","CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table([head] + data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ]))
    return tbl

# ----------------------------
# Walls structure
# ----------------------------
def create_walls_structure_table(walls_bd: dict, plan_data: dict) -> Table:
    items = walls_bd.get("items", [])
    head = [P("Tip Perete","CellBold"), P("Suprafață","CellBold"), P("Preț/m²","CellBold"), P("Total","CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Perete")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(name,"Cell"),
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = walls_bd.get("total_cost", 0.0)
    data.append([P("TOTAL","CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table([head] + data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ]))
    return tbl

# ----------------------------
# Floors & Ceilings
# ----------------------------
def create_floors_ceilings_table(floors_bd: dict) -> Table:
    items = floors_bd.get("items", [])
    head = [P("Element","CellBold"), P("Suprafață","CellBold"), P("Preț/m²","CellBold"), P("Total","CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Planșeu/Tavan")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(name,"Cell"),
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = floors_bd.get("total_cost", 0.0)
    data.append([P("TOTAL","CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table([head] + data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ]))
    return tbl

# ----------------------------
# Roof detailed
# ----------------------------
def create_roof_detailed_table(roof_bd: dict) -> Table:
    items = roof_bd.get("items", []) or roof_bd.get("detailed_items", [])
    head = [P("Componentă","CellBold"), P("Detalii","CellBold"), P("Cost","CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", it.get("category", "—"))
        details = it.get("details", "")
        cost = float(it.get("cost", it.get("total_cost", 0.0)))
        
        data.append([
            P(name,"Cell"),
            P(details if details else "—","CellSmall"),
            P(format_money(cost),"CellBold")
        ])
    
    total_cost = roof_bd.get("total_cost", 0.0)
    data.append([P("TOTAL ACOPERIȘ","CellBold"), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table([head] + data, colWidths=[50*mm, 75*mm, 40*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(2,1),(2,-1),"RIGHT"),
    ]))
    return tbl

# ----------------------------
# Openings detailed (doors & windows)
# ----------------------------
def create_openings_detailed_table(openings_bd: dict) -> Table:
    items = openings_bd.get("items", [])
    head = [
        P("ID","CellBold"), P("Tip","CellBold"), P("Locație","CellBold"),
        P("Material","CellBold"), P("Dim. (m)","CellBold"),
        P("Arie","CellBold"), P("Preț/m²","CellBold"), P("Total","CellBold")
    ]
    data = []
    
    for it in items:
        item_id = str(it.get("id", "—"))
        item_type = str(it.get("type", "—")).replace("_", " ").title()
        location = it.get("location", "—")
        material = it.get("material", "—")
        dims = it.get("dimensions_m", "—")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        
        data.append([
            P(item_id,"CellSmall"),
            P(item_type,"Cell"),
            P(location,"CellSmall"),
            P(material,"CellSmall"),
            P(dims,"CellSmall"),
            P(f"{area:.2f}","Cell"),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = openings_bd.get("total_cost", 0.0)
    data.append([
        P("","Cell"), P("TOTAL","CellBold"), P(""), P(""), P(""), P(""), P(""),
        P(format_money(total_cost),"CellBold")
    ])
    
    tbl = Table([head] + data, colWidths=[10*mm, 25*mm, 20*mm, 20*mm, 22*mm, 18*mm, 22*mm, 28*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(5,1),(-1,-1),"RIGHT"),
        ("FONTSIZE",(0,1),(-1,-1), 8),
    ]))
    return tbl

# ----------------------------
# Finishes
# ----------------------------
def create_finishes_table(finishes_bd: dict) -> Table:
    items = finishes_bd.get("items", [])
    head = [P("Finisaj","CellBold"), P("Suprafață","CellBold"), P("Preț/m²","CellBold"), P("Total","CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Finisaj")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(name,"Cell"),
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = finishes_bd.get("total_cost", 0.0)
    data.append([P("TOTAL","CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table([head] + data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
    ]))
    return tbl

# ----------------------------
# Utilities detailed
# ----------------------------
def create_utilities_detailed_table(utilities_bd: dict) -> Table:
    items = utilities_bd.get("items", []) or utilities_bd.get("detailed_items", [])
    head = [
        P("Categorie","CellBold"), P("Suprafață","CellBold"),
        P("Preț Bază/m²","CellBold"), P("Modif.","CellBold"),
        P("Preț Final/m²","CellBold"), P("Total","CellBold")
    ]
    data = []
    
    for it in items:
        cat = it.get("category", it.get("name", "—"))
        cat_label = {
            "electricity": "Electricitate",
            "heating": "Încălzire",
            "ventilation": "Ventilație",
            "sewage": "Canalizare"
        }.get(cat, cat)
        
        area = float(it.get("area_m2", 0.0))
        base = float(it.get("base_price_per_m2", 0.0))
        
        # Modifiers
        mods = []
        if "type_modifier" in it:
            mods.append(f"Tip:{it['type_modifier']:.2f}")
        if "energy_modifier" in it:
            mods.append(f"Ener:{it['energy_modifier']:.2f}")
        mod_str = ", ".join(mods) if mods else "—"
        
        final = float(it.get("final_price_per_m2", base))
        total = float(it.get("total_cost", 0.0))
        
        data.append([
            P(cat_label,"Cell"),
            P(format_area(area)),
            P(format_money(base),"CellSmall"),
            P(mod_str,"CellSmall"),
            P(format_money(final),"Cell"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = utilities_bd.get("total_cost", 0.0)
    data.append([
        P("TOTAL","CellBold"), P(""), P(""), P(""), P(""),
        P(format_money(total_cost),"CellBold")
    ])
    
    tbl = Table([head] + data, colWidths=[30*mm, 25*mm, 25*mm, 25*mm, 27*mm, 30*mm])
    tbl.setStyle(TableStyle([
        ("GRID",(0,0),(-1,-1),0.3,colors.black),
        ("BACKGROUND",(0,0),(-1,0), COLORS["bg_header"]),
        ("BACKGROUND",(0,-1),(-1,-1), COLORS["bg_light"]),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
        ("ALIGN",(1,1),(-1,-1),"RIGHT"),
        ("FONTSIZE",(0,1),(-1,-1), 8),
    ]))
    return tbl

# ----------------------------
# Final totals (multi-plan)
# ----------------------------
def create_totals_summary_table(totals: dict) -> Table:
    rows = [[P("Categorie","CellBold"), P("Total (EUR)","CellBold")]]
    items = [
        ("Fundație", totals.get("foundation", 0.0)),
        ("Structură Pereți", totals.get("structure", 0.0)),
        ("Planșee & Tavane", totals.get("floors_ceilings", 0.0)),
        ("Acoperiș", totals.get("roof", 0.0)),
        ("Tâmplărie (Uși & Ferestre)", totals.get("openings", 0.0)),
        ("Finisaje", totals.get("finishes", 0.0)),
        ("Utilități & Instalații", totals.get("utilities", 0.0)),
    ]
    subtotal = sum(v for _, v in items if v)

    for label, value in items:
        if value > 0:
            rows.append([P(label, "Cell"), P(format_money(value), "CellBold")])

    rows.append([P("SUBTOTAL","CellBold"), P(format_money(subtotal),"CellBold")])

    vat = subtotal * 0.19
    total_with_vat = subtotal + vat
    rows.append([P("TVA (19%)","Cell"), P(format_money(vat),"Cell")])
    rows.append([P("TOTAL FINAL (cu TVA)","CellBold"), P(format_money(total_with_vat),"CellBold")])

    table = Table(rows, colWidths=[100*mm, 60*mm])
    table.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.5, COLORS["border"]),
        ("BACKGROUND", (0,0), (-1,0), COLORS["bg_header"]),
        ("BACKGROUND", (0,-3), (-1,-3), COLORS["bg_light"]),
        ("BACKGROUND", (0,-1), (-1,-1), COLORS["success"]),
        ("TEXTCOLOR", (0,-1), (-1,-1), colors.white),
        ("ALIGN", (1,1), (1,-1), "RIGHT"),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,-1), (-1,-1), 11),
        ("FONTNAME", (0,-1), (-1,-1), BOLD_FONT),
    ]))
    return table