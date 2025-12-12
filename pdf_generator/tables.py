# tables.py - Toate funcțiile de creare tabele pentru PDF
from reportlab.platypus import Table, TableStyle, Paragraph
from reportlab.lib import colors
from reportlab.lib.units import mm

from .styles import get_styles, COLORS, BOLD_FONT, BASE_FONT
from .utils import format_money, format_area, format_length, safe_get

# Placeholder for GermanEnforcer (used only for type hinting/reference)
GermanEnforcer = object 

def P(text: str, style_name: str = "Cell") -> Paragraph:
    styles = get_styles()
    return Paragraph(str(text).replace("\n", "<br/>"), styles[style_name])

# ----------------------------
# Client info
# ----------------------------
def create_client_info_table(client_data: dict, enforcer: GermanEnforcer) -> Table:
    rows = [
        [P(enforcer.get("Client"), "CellBold"), P(enforcer.get(client_data.get("nume", "—")), "Cell")],
        [P(enforcer.get("Telefon"), "CellBold"), P(client_data.get("telefon", "—"), "Cell")],
        [P(enforcer.get("Email"), "CellBold"), P(client_data.get("email", "—"), "Cell")],
        [P(enforcer.get("Localitate"), "CellBold"), P(enforcer.get(client_data.get("localitate", "—")), "Cell")],
        [P(enforcer.get("Proiect"), "CellBold"), P(enforcer.get(client_data.get("referinta", "—")), "Cell")],
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
def create_inputs_info_table(inputs: dict, enforcer: GermanEnforcer) -> Table:
    """Tabel cu toate inputurile și caracteristicile proiectului"""
    # Traducem etichetele hardcodate
    label_map = {
        "tipSistem": enforcer.get("Sistem constructiv"),
        "gradPrefabricare": enforcer.get("Grad prefabricare"),
        "tipFundatie": enforcer.get("Tip fundație"),
        "tipAcoperis": enforcer.get("Tip acoperiș"),
        "nivelOferta": enforcer.get("Nivel ofertă"),
        "finisajInterior": enforcer.get("Finisaj interior"),
        "fatada": enforcer.get("Finisaj exterior"),
        "tamplarie": enforcer.get("Tâmplărie"),
        "materialAcoperis": enforcer.get("Material acoperiș"),
        "nivelEnergetic": enforcer.get("Nivel energetic"),
        "incalzire": enforcer.get("Tip încălzire"),
        "ventilatie": enforcer.get("Ventilație"),
    }
    
    rows = [[P(enforcer.get("Caracteristică"), "CellBold"), P(enforcer.get("Valoare"), "CellBold")]]
    
    for key, label in label_map.items():
        if key in inputs:
            val = inputs[key]
            if isinstance(val, bool):
                # Traducem valorile boolean "Da" / "Nu"
                val = enforcer.get("Da") if val else enforcer.get("Nu")
            else:
                # Traducem valorile input-urilor
                val = enforcer.get(str(val))
            rows.append([P(label, "Cell"), P(val, "Cell")])
    
    # Adaugă și alte chei necunoscute (traducem cheia și valoarea)
    for k, v in inputs.items():
        if k not in label_map and v:
            rows.append([P(enforcer.get(str(k)), "CellSmall"), P(enforcer.get(str(v)), "CellSmall")])

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
def create_plan_summary_table(plan_data: dict, plan_id: str, enforcer: GermanEnforcer) -> Table:
    floor_type = safe_get(plan_data, "floor_type", default="unknown")
    house_area = safe_get(plan_data, "house_area_m2", default=0.0)

    # Traducem denumirile etajelor
    floor_names = {
        "ground_floor": enforcer.get("Parter (Ground Floor)"),
        "top_floor": enforcer.get("Etaj (Top Floor)"),
        "intermediate": enforcer.get("Etaj Intermediar"),
        "unknown": enforcer.get("Necunoscut"),
    }

    rows = [
        [P(enforcer.get("Plan ID"), "CellBold"), P(enforcer.get(plan_id), "Cell")],
        [P(enforcer.get("Tip Etaj"), "CellBold"), P(floor_names.get(floor_type, enforcer.get(floor_type)), "Cell")],
        [P(enforcer.get("Suprafață Totală"), "CellBold"), P(format_area(house_area), "Cell")],
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
def create_construction_surfaces_table(pricing_data: dict, plan_data: dict, enforcer: GermanEnforcer) -> Table:
    """Tabel similar cu 'Konstruktionen & Oberflächen' din offer_pdf.py vechi"""
    bd = pricing_data.get("breakdown", {})
    
    # Structuri pereți (logica rămâne neschimbată)
    walls_items = bd.get("structure_walls", {}).get("items", [])
    int_area = 0.0; ext_area = 0.0; int_cost = 0.0; ext_cost = 0.0
    for it in walls_items:
        name = it.get("name", "").lower()
        area = float(it.get("area_m2", 0.0))
        cost = float(it.get("total_cost", 0.0))
        if "interior" in name: int_area += area; int_cost += cost
        elif "exterior" in name: ext_area += area; ext_cost += cost
    
    # Planșee & Tavane (logica rămâne neschimbată)
    fc_items = bd.get("floors_ceilings", {}).get("items", [])
    floor_area = 0.0; floor_cost = 0.0; ceiling_area = 0.0; ceiling_cost = 0.0
    for it in fc_items:
        name = it.get("name", "").lower()
        area = float(it.get("area_m2", 0.0))
        cost = float(it.get("total_cost", 0.0))
        if "podea" in name or "planșeu" in name: floor_area += area; floor_cost += cost
        if "tavan" in name: ceiling_area += area; ceiling_cost += cost
    
    # Fundație (logica rămâne neschimbată)
    found_items = bd.get("foundation", {}).get("items", [])
    found_area = 0.0; found_cost = 0.0
    if found_items:
        found_area = float(found_items[0].get("area_m2", 0.0))
        found_cost = float(found_items[0].get("total_cost", 0.0))
    
    # Traducerea etichetelor coloanelor
    head = [P(enforcer.get("Element"),"CellBold"), P(enforcer.get("Suprafață"),"CellBold"), P(enforcer.get("Preț/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")]
    data = []
    
    if found_cost > 0:
        data.append([
            P(enforcer.get("Fundație / Placă"),"Cell"),
            P(format_area(found_area)),
            P(format_money(found_cost/found_area if found_area else 0),"CellSmall"),
            P(format_money(found_cost),"CellBold")
        ])
    
    if floor_cost > 0:
        data.append([
            P(enforcer.get("Structură Planșeu"),"Cell"),
            P(format_area(floor_area)),
            P(format_money(floor_cost/floor_area if floor_area else 0),"CellSmall"),
            P(format_money(floor_cost),"CellBold")
        ])
    
    if int_cost > 0:
        data.append([
            P(enforcer.get("Pereți Interiori – Structură"),"Cell"),
            P(format_area(int_area)),
            P(format_money(int_cost/int_area if int_area else 0),"CellSmall"),
            P(format_money(int_cost),"CellBold")
        ])
    
    if ext_cost > 0:
        data.append([
            P(enforcer.get("Pereți Exteriori – Structură"),"Cell"),
            P(format_area(ext_area)),
            P(format_money(ext_cost/ext_area if ext_area else 0),"CellSmall"),
            P(format_money(ext_cost),"CellBold")
        ])
    
    if ceiling_cost > 0:
        data.append([
            P(enforcer.get("Structură Tavan"),"Cell"),
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
                P(enforcer.get(name),"Cell"), # Traducem numele finisajului
                P(format_area(area)),
                P(format_money(cost/area if area else 0),"CellSmall"),
                P(format_money(cost),"CellBold")
            ])
    
    if not data:
        data = [[P(enforcer.get("—")), P(enforcer.get("—")), P(enforcer.get("—")), P(enforcer.get("—"))]]
    
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
def create_foundation_table(foundation_bd: dict, enforcer: GermanEnforcer) -> Table:
    items = foundation_bd.get("items", [])
    head = [P(enforcer.get("Descriere"),"CellBold"), P(enforcer.get("Suprafață"),"CellBold"), P(enforcer.get("Preț/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Fundație")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(enforcer.get(name),"Cell"), # Traducem numele elementului
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = foundation_bd.get("total_cost", 0.0)
    data.append([P(enforcer.get("TOTAL"),"CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table(data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
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
def create_walls_structure_table(walls_bd: dict, plan_data: dict, enforcer: GermanEnforcer) -> Table:
    items = walls_bd.get("items", [])
    head = [P(enforcer.get("Tip Perete"),"CellBold"), P(enforcer.get("Suprafață"),"CellBold"), P(enforcer.get("Preț/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Perete")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(enforcer.get(name),"Cell"), # Traducem numele elementului
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = walls_bd.get("total_cost", 0.0)
    data.append([P(enforcer.get("TOTAL"),"CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table(data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
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
def create_floors_ceilings_table(floors_bd: dict, enforcer: GermanEnforcer) -> Table:
    items = floors_bd.get("items", [])
    head = [P(enforcer.get("Element"),"CellBold"), P(enforcer.get("Suprafață"),"CellBold"), P(enforcer.get("Preț/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Planșeu/Tavan")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(enforcer.get(name),"Cell"), # Traducem numele elementului
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = floors_bd.get("total_cost", 0.0)
    data.append([P(enforcer.get("TOTAL"),"CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table(data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
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
def create_roof_detailed_table(roof_bd: dict, enforcer: GermanEnforcer) -> Table:
    items = roof_bd.get("items", []) or roof_bd.get("detailed_items", [])
    head = [P(enforcer.get("Componentă"),"CellBold"), P(enforcer.get("Detalii"),"CellBold"), P(enforcer.get("Cost"),"CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", it.get("category", "—"))
        details = it.get("details", "")
        cost = float(it.get("cost", it.get("total_cost", 0.0)))
        
        data.append([
            P(enforcer.get(name),"Cell"),
            P(enforcer.get(details) if details else enforcer.get("—"),"CellSmall"), # Traducem detaliile
            P(format_money(cost),"CellBold")
        ])
    
    total_cost = roof_bd.get("total_cost", 0.0)
    data.append([P(enforcer.get("TOTAL ACOPERIȘ"),"CellBold"), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table(data, colWidths=[50*mm, 75*mm, 40*mm])
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
def create_openings_detailed_table(openings_bd: dict, enforcer: GermanEnforcer) -> Table:
    items = openings_bd.get("items", [])
    head = [
        P(enforcer.get("ID"),"CellBold"), P(enforcer.get("Tip"),"CellBold"), P(enforcer.get("Locație"),"CellBold"),
        P(enforcer.get("Material"),"CellBold"), P(enforcer.get("Dim. (m)"),"CellBold"),
        P(enforcer.get("Arie"),"CellBold"), P(enforcer.get("Preț/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")
    ]
    data = []
    
    for it in items:
        item_id = str(it.get("id", "—"))
        # Traducem tipul, locația și materialul
        item_type = enforcer.get(str(it.get("type", "—")).replace("_", " ").title())
        location = enforcer.get(it.get("location", "—"))
        material = enforcer.get(it.get("material", "—"))
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
        P("","Cell"), P(enforcer.get("TOTAL"),"CellBold"), P(""), P(""), P(""), P(""), P(""),
        P(format_money(total_cost),"CellBold")
    ])
    
    tbl = Table(data, colWidths=[10*mm, 25*mm, 20*mm, 20*mm, 22*mm, 18*mm, 22*mm, 28*mm])
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
def create_finishes_table(finishes_bd: dict, enforcer: GermanEnforcer) -> Table:
    items = finishes_bd.get("items", [])
    head = [P(enforcer.get("Finisaj"),"CellBold"), P(enforcer.get("Suprafață"),"CellBold"), P(enforcer.get("Preț/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")]
    data = []
    
    for it in items:
        name = it.get("name", "Finisaj")
        area = float(it.get("area_m2", 0.0))
        unit = float(it.get("unit_price", 0.0))
        total = float(it.get("total_cost", 0.0))
        data.append([
            P(enforcer.get(name),"Cell"),
            P(format_area(area)),
            P(format_money(unit),"CellSmall"),
            P(format_money(total),"CellBold")
        ])
    
    total_cost = finishes_bd.get("total_cost", 0.0)
    data.append([P(enforcer.get("TOTAL"),"CellBold"), P(""), P(""), P(format_money(total_cost),"CellBold")])
    
    tbl = Table(data, colWidths=[60*mm, 35*mm, 30*mm, 35*mm])
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
def create_utilities_detailed_table(utilities_bd: dict, enforcer: GermanEnforcer) -> Table:
    items = utilities_bd.get("items", []) or utilities_bd.get("detailed_items", [])
    head = [
        P(enforcer.get("Categorie"),"CellBold"), P(enforcer.get("Suprafață"),"CellBold"),
        P(enforcer.get("Preț Bază/m²"),"CellBold"), P(enforcer.get("Modif."),"CellBold"),
        P(enforcer.get("Preț Final/m²"),"CellBold"), P(enforcer.get("Total"),"CellBold")
    ]
    data = []
    
    for it in items:
        cat = it.get("category", it.get("name", "—"))
        # Traducerea etichetei categoriei locale
        cat_label = {
            "electricity": enforcer.get("Electricitate"),
            "heating": enforcer.get("Încălzire"),
            "ventilation": enforcer.get("Ventilație"),
            "sewage": enforcer.get("Canalizare")
        }.get(cat, enforcer.get(cat))
        
        area = float(it.get("area_m2", 0.0))
        base = float(it.get("base_price_per_m2", 0.0))
        
        # Modifiers
        mods = []
        if "type_modifier" in it:
            mods.append(f"{enforcer.get('Tip')}:{it['type_modifier']:.2f}")
        if "energy_modifier" in it:
            mods.append(f"{enforcer.get('Ener')}:{it['energy_modifier']:.2f}")
        mod_str = ", ".join(mods) if mods else enforcer.get("—")
        
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
        P(enforcer.get("TOTAL"),"CellBold"), P(""), P(""), P(""), P(""),
        P(format_money(total_cost),"CellBold")
    ])
    
    tbl = Table(data, colWidths=[30*mm, 25*mm, 25*mm, 25*mm, 27*mm, 30*mm])
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
def create_totals_summary_table(totals: dict, enforcer: GermanEnforcer) -> Table:
    rows = [[P(enforcer.get("Categorie"),"CellBold"), P(enforcer.get("Total (EUR)"),"CellBold")]]
    items = [
        (enforcer.get("Fundație"), totals.get("foundation", 0.0)),
        (enforcer.get("Structură Pereți"), totals.get("structure", 0.0)),
        (enforcer.get("Planșee & Tavane"), totals.get("floors_ceilings", 0.0)),
        (enforcer.get("Acoperiș"), totals.get("roof", 0.0)),
        (enforcer.get("Tâmplărie (Uși & Ferestre)"), totals.get("openings", 0.0)),
        (enforcer.get("Finisaje"), totals.get("finishes", 0.0)),
        (enforcer.get("Utilități & Instalații"), totals.get("utilities", 0.0)),
    ]
    subtotal = sum(v for _, v in items if v)

    for label, value in items:
        if value > 0:
            rows.append([P(label, "Cell"), P(format_money(value), "CellBold")])

    rows.append([P(enforcer.get("SUBTOTAL"),"CellBold"), P(format_money(subtotal),"CellBold")])

    vat = subtotal * 0.19
    total_with_vat = subtotal + vat
    rows.append([P(enforcer.get("TVA (19%)"),"Cell"), P(format_money(vat),"Cell")])
    rows.append([P(enforcer.get("TOTAL FINAL (cu TVA)"),"CellBold"), P(format_money(total_with_vat),"CellBold")])

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