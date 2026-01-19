from __future__ import annotations
import json
import io
import os
import re
from pathlib import Path
from datetime import datetime
import random

# --- NEW: OpenAI Import ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase.pdfmetrics import stringWidth

from PIL import Image as PILImage, ImageEnhance, ImageOps

from config.settings import load_plan_infos, PlansListError, RUNNER_ROOT, PROJECT_ROOT
from config.frontend_loader import load_frontend_data_for_run
from branding.db_loader import fetch_tenant_branding

# ---------- 1. DICTIONAR STATIC EXTINS (CONSTANTE) ----------
STATIC_TRANSLATIONS = {
    # Unități
    "buc": "Stk.",
    "buc.": "Stk.",
    "bucata": "Stk.",
    "piece": "Stk.",
    "pieces": "Stk.",
    "mp": "m²",
    "m2": "m²",
    "ml": "m",
    "kg": "kg",
    "tone": "t",
    "ora": "Std.",
    "manopera": "Arbeitsleistung",
    
    # Structură & Nivele
    "Parter": "Erdgeschoss",
    "Ground Floor": "Erdgeschoss",
    "ground_floor": "Erdgeschoss",
    "Etaj": "Obergeschoss",
    "top_floor": "Obergeschoss / Dachgeschoss",
    "Etaj 1": "1. Obergeschoss",
    "Mansarda": "Dachgeschoss",
    "Acoperis": "Dach",
    "Dachfläche": "Dachfläche",
    "Fundație": "Fundament",
    "Fundament (Sockel)": "Fundament (Sockel)",
    "Placa": "Bodenplatte",
    
    # ELEMENTE SCAPATE ANTERIOR (FIX)
    "Structură Tavan": "Deckenkonstruktion",
    "Structura Tavan": "Deckenkonstruktion",
    "Bodenstruktur": "Bodenaufbau",
    "Deckenstruktur": "Deckenaufbau",
    "Structura Podea": "Bodenkonstruktion",
    
    # Elemente Constructive
    "Pereti": "Wände",
    "Pereti Exteriori": "Außenverkleidung",
    "Pereti Interiori": "Innenverkleidung",
    "Planseu": "Geschossdecke",
    "Grinda": "Holzbalken",
    "Stalp": "Stütze",
    "Scara": "Treppe",
    "Balustrada": "Geländer",
    
    # Materiale
    "Beton": "Beton",
    "Lemn": "Holz",
    "Caramida": "Ziegel",
    "Fier": "Stahl",
    "Vata": "Dämmung",
    "Rigips": "Gipskarton",
    
    # Categorii Oferta
    "Structură": "Rohbau / Konstruktion",
    "Arhitectura": "Architektur",
    "Instalatii": "Haustechnik",
    "Finisaje": "Ausbau & Oberflächen",
    "Casă completă": "Schlüsselfertig",
    "La rosu": "Rohbau",
    
    # Utilități
    "Electrice": "Elektroinstallation",
    "Sanitare": "Sanitärinstallation",
    "Termice": "Heizungstechnik",
    "Canalizare": "Abwasser",
    
    # Deschideri
    "Fereastra": "Fenster",
    "Usa": "Tür",
    "Usa intrare": "Haustür",
    "Dublu": "Zweiflügelig",
    "Simplu": "Einflügelig",
    "Double": "Zweiflügelig",
    "Single": "Einflügelig",
    
    # Diverse & Hardcoded Labels (Extins)
    "Total": "Gesamt",
    "Pret": "Preis",
    "Cantitate": "Menge",
    "Descriere": "Beschreibung",
    "Angebot": "Angebot", 
    "Nr.": "Nr.",
    "Bauherr / Kunde": "Bauherr / Kunde",
    "Ort / Bauort": "Ort / Bauort",
    "Telefon": "Telefon",
    "E-Mail": "E-Mail",
    "Bauvorhaben": "Bauvorhaben",
    "Angebot für Ihr Chiemgauer Massivholzhaus": "Angebot für Ihr Chiemgauer Massivholzhaus",
    "Sehr geehrte Damen und Herren,": "Liebe Kundschaft,", 
    "vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.": "vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung.",
    "HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.": "HINWEIS: Unverbindliche Kostenschätzung. Kein verbindliches Angebot.",
    "Planungsebene": "Planungsebene",
    "Gesamtkostenzusammenstellung": "Gesamtkostenzusammenstellung",
    "Baukosten (Konstruktion, Ausbau, Technik)": "Baukosten (Konstruktion, Ausbau, Technik)",
    "Baustelleneinrichtung, Logistik & Planung (10%)": "Baustelleneinrichtung, Logistik & Planung",
    "Bauleitung, Koordination & Gewinn (10%)": "Bauleitung & Koordination",
    "Nettosumme (exkl. MwSt.)": "Nettosumme (exkl. MwSt.)",
    "MwSt. (19%)": "MwSt. (19%)",
    "GESAMTSUMME BRUTTO": "GESAMTSUMME BRUTTO",
    "Position": "Position",
    "Betrag": "Betrag",
    "Annahmen & Vorbehalte": "Annahmen & Vorbehalte",
    "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.": "Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen.",
    "Bauteil": "Bauteil", 
    "Fläche": "Fläche", 
    "Preis/m²": "Preis/m²", 
    "Gesamt": "Gesamt", 
    "SUMME": "SUMME",
    "Dachkonstruktion – Detail": "Dachkonstruktion – Detail", 
    "Komponente": "Komponente", 
    "Bemerkung": "Bemerkung", 
    "Menge": "Menge", 
    "Preis": "Preis", 
    "SUMME DACH": "SUMME DACH",
    "Treppenanlagen": "Treppenanlagen", 
    "Beschreibung": "Beschreibung", 
    "SUMME TREPPEN": "SUMME TREPPEN", 
    "Zusammenfassung Fenster & Türen": "Zusammenfassung Fenster & Türen",
    "Stückzahl": "Stückzahl", 
    "Ø Preis/Stk.": "Ø Preis/Stk.", 
    "SUMME ÖFFNUNGEN": "SUMME ÖFFNUNGEN", 
    "Zusammenfassung Haustechnik & Installationen": "Zusammenfassung Haustechnik & Installationen",
    "Gewerk / Kategorie": "Gewerk / Kategorie", 
    "Gesamtpreis": "Gesamtpreis", 
    "SUMME HAUSTECHNIK": "SUMME HAUSTECHNIK", 
    "Außentür": "Außentür", 
    "Innentür": "Innentür", 
    "Fenster": "Fenster",
    "Türen": "Türen",
    "Zweiflügelig": "Zweiflügelig", 
    "Einflügelig": "Einflügelig",
    "Fundament / Bodenplatte": "Fundament / Bodenplatte",
    "Tragwerkskonstruktion – Wände": "Tragwerkskonstruktion – Wände",
    "Geschossdecken & Balken": "Geschossdecken & Balken",
    "Oberflächen & Ausbau": "Oberflächen & Ausbau",
    "Kategorie": "Kategorie",
    
    # Admin PDF translations
    "Ventilație": "Ventilation",
    "Măsurători Pereți": "Wandmaße",
    # Wall measurements table translations
    "Tip Perete": "Wandtyp",
    "Utilizare": "Verwendung",
    "Lungime (m)": "Länge (m)",
    "Arie Brută (m²)": "Bruttofläche (m²)",
    "Deschideri (m²)": "Öffnungen (m²)",
    "Arie Netă (m²)": "Nettofläche (m²)",
    "Pereți Exteriori": "Außenwände",
    "Pereți Interiori": "Innenwände",
    "Structură (Skeleton)": "Struktur (Skeleton)",
    "Finisaje (Outline)": "Ausbau",
    # Note: "Structură" and "Finisaje" are used in wall measurements table with specific translations
    # For wall measurements context: "Structură" = "Struktur", "Finisaje" = "Ausbau"
    # These are handled directly in the table code to avoid conflicts with general translations
    
    # NOU: Textele de extins
    "Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen.": "Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen.",
    "Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung.": "Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung.",
    
    # Secțiuni noi pentru PDF simplificat
    "Klärung des Zwecks (richtige Erwartungen setzen)": "Klärung des Zwecks (richtige Erwartungen setzen)",
    "Diese Schätzung dient der Orientierung und ist für die erste Diskussion mit dem Auftraggeber bestimmt.": "Diese Schätzung dient der Orientierung und ist für die erste Diskussion mit dem Auftraggeber bestimmt.",
    "Die Schätzung basiert auf den bereitgestellten Informationen und auf der automatischen Analyse der Pläne.": "Die Schätzung basiert auf den bereitgestellten Informationen und auf der automatischen Analyse der Pläne.",
    "Das Dokument stellt kein verbindliches Angebot dar, sondern hilft bei:": "Das Dokument stellt kein verbindliches Angebot dar, sondern hilft bei:",
    "schnelles Erhalten eines realistischen Budgetüberblicks": "schnelles Erhalten eines realistischen Budgetüberblicks",
    "Vermeidung von Zeitverlust bei Projekten, die finanziell nicht machbar sind": "Vermeidung von Zeitverlust bei Projekten, die finanziell nicht machbar sind",
    "Projektübersicht (leicht verständlich)": "Projektübersicht (leicht verständlich)",
    "Projektübersicht": "Projektübersicht",
    "Allgemeine Baudaten (Auszug):": "Allgemeine Baudaten (Auszug):",
    "Nutzfläche": "Nutzfläche",
    "Anzahl der Ebenen": "Anzahl der Ebenen",
    "Anzahl der Stockwerke": "Anzahl der Stockwerke",
    "Bausystem": "Bausystem",
    "Dachtyp": "Dachtyp",
    "Dachmaterial": "Dachmaterial",
    "Heizsystem": "Heizsystem",
    "Fertigstellungsgrad": "Fertigstellungsgrad",
    "gemäß verfügbaren Plänen und Informationen": "gemäß verfügbaren Plänen und Informationen",
    "Kostenstruktur (vereinfacht, klar)": "Kostenstruktur (vereinfacht, klar)",
    "Kostenstruktur": "Kostenstruktur",
    "Komponente": "Komponente",
    "Geschätzte Kosten": "Geschätzte Kosten",
    "Hausstruktur (Wände, Decken, Dach)": "Hausstruktur (Wände, Decken, Dach)",
    "Fenster & Türen": "Fenster & Türen",
    "Innenausbau": "Innenausbau",
    "Installationen & Technik": "Installationen & Technik",
    "GESAMT": "GESAMT",
    "Was NICHT enthalten ist (sehr wichtig)": "Was NICHT enthalten ist (sehr wichtig)",
    "Nicht in dieser Schätzung enthalten:": "Nicht in dieser Schätzung enthalten:",
    "Grundstückskosten": "Grundstückskosten",
    "Außenanlagen (Zaun, Hof, Wege)": "Außenanlagen (Zaun, Hof, Wege)",
    "Küche und Möbel": "Küche und Möbel",
    "Sonderausstattung oder individuelle Anforderungen": "Sonderausstattung oder individuelle Anforderungen",
    "Steuern, Genehmigungen, Anschlüsse": "Steuern, Genehmigungen, Anschlüsse",
    "Diese werden in den nächsten Phasen besprochen.": "Diese werden in den nächsten Phasen besprochen.",
    "Genauigkeit der Schätzung (rechtliche Sicherheit)": "Genauigkeit der Schätzung (rechtliche Sicherheit)",
    "Die Schätzung liegt erfahrungsgemäß in einem Bereich von ±10-15 %, abhängig von den finalen Ausführungs- und Planungsdetails.": "Die Schätzung liegt erfahrungsgemäß in einem Bereich von ±10-15 %, abhängig von den finalen Ausführungs- und Planungsdetails.",
    "Rechtlicher Hinweis / Haftungsausschluss": "Rechtlicher Hinweis / Haftungsausschluss",
    "Dieses Dokument ist eine unverbindliche Kostenschätzung zur ersten Budgetorientierung und ersetzt kein verbindliches Angebot.": "Dieses Dokument ist eine unverbindliche Kostenschätzung zur ersten Budgetorientierung und ersetzt kein verbindliches Angebot.",
    "Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma.": "Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma.",
    "Abweichungen durch Planänderungen, Ausführungsdetails, Grundstücksgegebenheiten, behördliche Auflagen oder individuelle Wünsche sind möglich.": "Abweichungen durch Planänderungen, Ausführungsdetails, Grundstücksgegebenheiten, behördliche Auflagen oder individuelle Wünsche sind möglich.",
    "Nicht Bestandteil dieser Schätzung sind insbesondere:": "Nicht Bestandteil dieser Schätzung sind insbesondere:",
    "Außenanlagen (z. B. Einfriedungen, Einfahrten, Garten- und Landschaftsgestaltung)": "Außenanlagen (z. B. Einfriedungen, Einfahrten, Garten- und Landschaftsgestaltung)",
    "statische Berechnungen": "statische Berechnungen",
    "bauphysikalische Nachweise": "bauphysikalische Nachweise",
    "Grundstücks- und Bodenbeschaffenheit": "Grundstücks- und Bodenbeschaffenheit",
    "Förderungen, Gebühren und Abgaben": "Förderungen, Gebühren und Abgaben",
    "behördliche oder rechtliche Prüfungen": "behördliche oder rechtliche Prüfungen",
    "Die endgültige Preisfestlegung erfolgt ausschließlich im Rahmen eines individuellen Angebots nach detaillierter Planung und Prüfung durch die ausführende Holzbaufirma.": "Die endgültige Preisfestlegung erfolgt ausschließlich im Rahmen eines individuellen Angebots nach detaillierter Planung und Prüfung durch die ausführende Holzbaufirma.",
    "Planungsebenen": "Planungsebenen",
    
    # Traduceri pentru valorile din formular
    # Sistem constructiv
    "CLT": "CLT",
    "Holzrahmen": "Holzrahmen",
    "HOLZRAHMEN": "Holzrahmen",
    "Massivholz": "Massivholz",
    "Panouri": "Paneele",
    "Module": "Module",
    "Montaj pe șantier": "Montage auf der Baustelle",
    "Placă": "Platte",
    "Piloți": "Pfähle",
    "Soclu": "Sockel",
    "Drept": "Flachdach",
    "Două ape": "Satteldach",
    "Patru ape": "Walmdach",
    "Mansardat": "Mansarddach",
    "Șarpantă complexă": "Komplexes Dach",
    "Satteldach": "Satteldach",
    # Materiale finisaj
    "Tencuială": "Putz",
    "Lemn": "Holz",
    "Fibrociment": "Faserzement",
    "Mix": "Mix",
    "Lemn-Aluminiu": "Holz-Aluminium",
    "PVC": "PVC",
    "Aluminiu": "Aluminium",
    # Acoperiș
    "Țiglă": "Dachziegel",
    "Tablă": "Dachblech",
    "Membrană": "Dachmembran",
    # Performanță energetică
    "Standard": "Standard",
    "KfW 55": "KfW 55",
    "KfW 40": "KfW 40",
    "KfW 40+": "KfW 40+",
    "Gaz": "Gas",
    "Pompa de căldură": "Wärmepumpe",
    "Electric": "Elektrisch",
    "Kamin": "Kamin",
    "Kaminabzug": "Kaminabzug",
    "Kamin & Kaminabzug": "Kamin & Kaminabzug",
    "Ja": "Ja",
    "Treppe": "Treppe",
    # Nivel ofertă
    "Structură": "Rohbau / Konstruktion",
    "Structură + ferestre": "Rohbau + Fenster",
    "Rohbau/Tragwerk": "Rohbau / Tragwerk",
    "Tragwerk + Fenster": "Tragwerk + Fenster",
    "Schlüsselfertig": "Schlüsselfertig",
    "Schlüsselfertiges Haus": "Schlüsselfertiges Haus",
    # Număr ferestre și uși
    "Anzahl Fenster": "Anzahl Fenster",
    "Anzahl Türen": "Anzahl Türen",
    "detektiert": "detektiert",
    "Grad prefabricare": "Vorfabrizierungsgrad",
    "Tip fundație": "Fundamenttyp",
    "Sistem constructiv": "Bausystem",
    "Tip acoperiș": "Dachtyp",
    "Tip acoperis": "Dachtyp",
    "Nivel ofertă": "Gewünschter Angebotsumfang",
    "Nivel oferta": "Gewünschter Angebotsumfang",
    "Nivel de ofertă dorit": "Gewünschter Angebotsumfang",
    "Nivel de oferta dorit": "Gewünschter Angebotsumfang",
    # Termeni din formular care trebuie traduse
    "Blockbau": "Blockbau",
    "Holzrahmen": "Holzrahmen",
    "Massivholz": "Massivholz",
    "Placă": "Bodenplatte",
    "Piloți": "Pfähle",
    "Soclu": "Sockel",
    "Structură": "Rohbau / Konstruktion",
    "Structură + ferestre": "Rohbau + Fenster",
    "Casă completă": "Schlüsselfertig",
    "Ușor (camion 40t)": "Leicht (LKW 40t)",
    "Mediu": "Mittel",
    "Dificil": "Schwierig",
    "Plan": "Eben",
    "Pantă ușoară": "Leichte Hanglage",
    "Pantă mare": "Starke Hanglage",
    # Termeni pentru finisaje
    "Tencuială": "Putz",
    "Fibrociment": "Faserzement",
    "Mix": "Mischung",
    "Lemn-Aluminiu": "Holz-Aluminium",
    "PVC": "Kunststoff",
    "Aluminiu": "Aluminium",
    # Termeni pentru acoperiș
    "Țiglă": "Dachziegel",
    "Țiglă ceramică": "Tondachziegel",
    "Țiglă beton": "Betondachstein",
    "Tablă": "Blech",
    "Tablă fălțuită": "Stehfalzblech",
    "Șindrilă bituminoasă": "Bitumschindel",
    "Membrană": "Membranbahn",
    "Membrană PVC": "PVC-Bahn",
    "Hidroizolație bitum": "Bitumenabdichtung",
    # Termeni pentru încălzire
    "Gaz": "Gas",
    "Pompa de căldură": "Wärmepumpe",
    "Electric": "Elektrisch",
    # Termeni pentru nivel energetic
    "KfW 55": "KfW 55",
    "KfW 40": "KfW 40",
    "KfW 40+": "KfW 40+",
    # Termeni pentru grad prefabricare
    "Montaj pe șantier": "Montage auf der Baustelle",
    "Prefabricare parțială": "Teilweise Vorfertigung",
    "Prefabricare completă": "Vollständige Vorfertigung",
    # Termeni pentru tipuri de acoperiș
    "Drept": "Flachdach",
    "Două ape": "Satteldach",
    "Patru ape": "Walmdach",
    "Mansardat": "Mansarddach",
    "Șarpantă complexă": "Komplexes Dach",
    # Termeni pentru finisaje per etaj
    "Finisaj interior": "Innenausbau",
    "Fațadă": "Fassade",
    "Finisaj interior (Parter)": "Innenausbau (Erdgeschoss)",
    "Fațadă (Parter)": "Fassade (Erdgeschoss)",
    "Finisaj interior (Etaj 1)": "Innenausbau (1. Obergeschoss)",
    "Fațadă (Etaj 1)": "Fassade (1. Obergeschoss)",
    "Finisaj interior (Etaj 2)": "Innenausbau (2. Obergeschoss)",
    # Termeni pentru finisaje interioare și exterioare (în loc de pereți)
    "Innenverkleidung": "Innenverkleidung",
    "Außenverkleidung": "Außenverkleidung",
    "Fațadă (Etaj 2)": "Fassade (2. Obergeschoss)",
    "Finisaj interior (Etaj 3)": "Innenausbau (3. Obergeschoss)",
    "Fațadă (Etaj 3)": "Fassade (3. Obergeschoss)",
    "Finisaj interior (Mansardă)": "Innenausbau (Mansarde)",
    "Fațadă (Mansardă)": "Fassade (Mansarde)",
    "Finisaj interior (Beci)": "Innenausbau (Keller)",
    # Termeni pentru etaje
    "Erdgeschoss": "Erdgeschoss",
    "Obergeschoss": "Obergeschoss",
    "1. Obergeschoss": "1. Obergeschoss",
    "2. Obergeschoss": "2. Obergeschoss",
    "3. Obergeschoss": "3. Obergeschoss",
    "Dachgeschoss": "Dachgeschoss",
    "Mansardă": "Mansarde",
    "Planungsebene": "Planungsebene",
    "Finisaj interior": "Innenausbau",
    "Fațadă": "Fassade",
    "Fenster & Türen (Material)": "Fenster & Türen (Material)",
    "Nivel energetic": "Energiestandard",
    "Gesamtnutzfläche": "Hausfläche",
    "Hausfläche": "Hausfläche",
    "Allgemeine Projektinformationen": "Allgemeine Projektinformationen",
}

# ---------- 2. AI TRANSLATION SERVICE (AGRESIV) ----------
class GermanEnforcer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if OpenAI and api_key else None
        self.cache = STATIC_TRANSLATIONS.copy()

    def get(self, text):
        """Returnează traducerea din cache. Dacă nu există, returnează originalul."""
        if not text or not isinstance(text, str): 
            return text
        
        translated = self.cache.get(text)
        
        if translated and '(' in translated and ')' in translated and translated != text:
            translated = re.sub(r'\s*\([^)]*\)\s*', '', translated).strip()
            return translated

        return translated if translated else text

    def translate_table_batch(self, texts: list[str]) -> dict[str, str]:
        """
        Traduce un batch de texte DIRECT pentru tabele.
        Filtrează automat: > 3 caractere, nu e preț/număr.
        """
        if not self.client:
            return {t: t for t in texts}
        
        # Filtrează ce merită tradus
        to_translate = []
        for t in texts:
            if not isinstance(t, str):
                continue
            t_clean = t.strip()
            if len(t_clean) <= 3:  # ✅ Skip < 3 caractere
                continue
            if re.match(r'^[\d\.,\s€]+$', t_clean):  # ✅ Skip prețuri/numere
                continue
            to_translate.append(t_clean)
        
        if not to_translate:
            return {t: t for t in texts}
        
        print(f"🇩🇪 [TableTranslation] Translating {len(to_translate)} items...")
        
        # Verifică cache-ul întâi
        results = {}
        missing = []
        
        for text in to_translate:
            if text in self.cache:
                results[text] = self.cache[text]
            else:
                missing.append(text)
        
        # Traduce ce lipsește
        if missing:
            prompt = (
                "You are a professional technical translator for the German construction industry. "
                "Translate the following JSON list from Romanian/English to German. "
                "Keep technical precision (e.g. 'Parter' -> 'Erdgeschoss', 'Structura Tavan' -> 'Deckenkonstruktion', "
                "'Pereti Exteriori' -> 'Außenverkleidung', 'Lemn' -> 'Holz'). "
                "Output ONLY a valid JSON object where keys are input text and values are German translations."
            )
            
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(missing)}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                
                translations = json.loads(response.choices[0].message.content)
                self.cache.update(translations)
                results.update(translations)
                
                print(f"✅ [TableTranslation] Cached {len(translations)} new translations")
                
            except Exception as e:
                print(f"⚠️ [TableTranslation] Error: {e}")
                for text in missing:
                    results[text] = text
        
        # Returnează dicționar complet
        final_results = {}
        for t in texts:
            if isinstance(t, str):
                t_clean = t.strip()
                final_results[t] = results.get(t_clean, t)
            else:
                final_results[t] = t
        
        return final_results

# ---------- FONTS & ASSETS ----------
FONTS_DIR = Path(__file__).parent.parent / "pdf_assets" / "fonts"
FONT_REG = FONTS_DIR / "DejaVuSans.ttf"
FONT_BOLD = FONTS_DIR / "DejaVuSans-Bold.ttf"
BASE_FONT, BOLD_FONT = "DejaVuSans", "DejaVuSans-Bold"

try:
    if FONT_REG.exists() and FONT_BOLD.exists():
        pdfmetrics.registerFont(TTFont(BASE_FONT, str(FONT_REG)))
        pdfmetrics.registerFont(TTFont(BOLD_FONT, str(FONT_BOLD)))
except Exception:
    BASE_FONT, BOLD_FONT = "Helvetica", "Helvetica-Bold"

COMPANY = {
    "name": "Chiemgauer Holzhaus",
    "legal": "LSP Holzbau GmbH & Co KG",
    "addr_lines": ["Seiboldsdorfer Mühle 1a", "83278 Traunstein"],
    "phone": "+49 (0) 861 / 166 192 0",
    "fax":   "+49 (0) 861 / 166 192 20",
    "email": "info@chiemgauer-holzhaus.de",
    "web":   "www.chiemgauer-holzhaus.de",
    "footer_left":  "Chiemgauer Holzhaus\nLSP Holzbau GmbH & Co KG\nRegistergericht Traunstein HRA Nr. 7311\nGeschäftsführer Bernhard Oeggl",
    "footer_mid":   "LSP Verwaltungs GmbH\nPersönlich haftende Gesellschafterin\nRegistergericht Traunstein HRB Nr. 13146",
    "footer_right": "Volksbank Raiffeisenbank Oberbayern Südost eG\nKto.Nr. 7 313 640  ·  BLZ 710 900 00\nIBAN: DE81 7109 0000 0007 3136 40   BIC: GENODEF1BGL   USt-ID: DE131544091",
}

IMG_IDENTITY = PROJECT_ROOT / "offer_identity.png"
IMG_LOGOS = PROJECT_ROOT / "offer_logos.png"

def _apply_branding(tenant_slug: str | None) -> dict:
    """
    Returns an overlay dict with:
      - company overrides
      - assets overrides
      - offer_prefix / handler_name
    """
    if not tenant_slug:
        return {}
    try:
        return fetch_tenant_branding(tenant_slug) or {}
    except Exception as e:
        print(f"⚠️ [PDF] Failed to load branding for tenant '{tenant_slug}': {e}", flush=True)
        return {}

def _asset_path(filename: str | None) -> Path | None:
    if not filename:
        return None
    p = PROJECT_ROOT / filename
    return p if p.exists() else None

def _clean_multiline_text(v: object, *, max_lines: int = 6, max_line_len: int = 72) -> str:
    """
    Clean text coming from DB config for footer/header blocks.
    - Convert literal '\\n' into real newlines
    - Strip non-printable chars
    - Cap number of lines and line length to avoid footer blowups
    """
    if v is None:
        return ""
    s = str(v)
    # Convert escaped newlines from JSON/SQL inserts
    s = s.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    # Normalize actual CRLF too
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # Strip control chars except newline
    s = "".join(ch for ch in s if ch == "\n" or (32 <= ord(ch) < 127) or ch in "ÄÖÜäöüß€")
    lines = [ln.strip() for ln in s.split("\n") if ln.strip()]
    lines = lines[:max_lines]
    clipped: list[str] = []
    for ln in lines:
        if len(ln) > max_line_len:
            clipped.append(ln[: max_line_len - 1].rstrip() + "…")
        else:
            clipped.append(ln)
    return "\n".join(clipped)

# ---------- LOGIC ----------
def _get_offer_inclusions(nivel_oferta: str) -> dict:
    INCLUSIONS = {
        "Structură": {
            "foundation": True, "structure_walls": True, "roof": True, "floors_ceilings": True,
            "openings": False, "finishes": False, "utilities": False
        },
        "Structură + ferestre": {
            "foundation": True, "structure_walls": True, "roof": True, "floors_ceilings": True,
            "openings": True, "finishes": False, "utilities": False
        },
        "Casă completă": {
            "foundation": True, "structure_walls": True, "roof": True, "floors_ceilings": True,
            "openings": True, "finishes": True, "utilities": True
        }
    }
    return INCLUSIONS.get(nivel_oferta, INCLUSIONS["Casă completă"])

# ---------- STYLES ----------
def _styles():
    s = getSampleStyleSheet()
    s.add(ParagraphStyle(name="H1", fontName=BOLD_FONT, fontSize=12, leading=22, spaceAfter=10, spaceBefore=6))
    s.add(ParagraphStyle(name="H2", fontName=BOLD_FONT, fontSize=11, leading=14, spaceBefore=8, spaceAfter=4))
    s.add(ParagraphStyle(name="H3", fontName=BOLD_FONT, fontSize=10, leading=13, spaceBefore=4, spaceAfter=2))
    s.add(ParagraphStyle(name="Body", fontName=BASE_FONT, fontSize=10, leading=14, spaceAfter=3))
    s.add(ParagraphStyle(name="Small", fontName=BASE_FONT, fontSize=7.2, leading=9.2))
    s.add(ParagraphStyle(name="Disclaimer", fontName=BASE_FONT, fontSize=8.5, leading=11, textColor=colors.HexColor("#333333")))
    # Prevent mid-word breaks/hyphenation inside table cells.
    s.add(ParagraphStyle(name="Cell", fontName=BASE_FONT, fontSize=9.5, leading=12, splitLongWords=0))
    s.add(ParagraphStyle(name="CellBold", fontName=BOLD_FONT, fontSize=9.5, leading=12, splitLongWords=0))
    s.add(ParagraphStyle(name="CellSmall", fontName=BASE_FONT, fontSize=9, leading=11, splitLongWords=0))
    return s

def _pretty_label(v: str) -> str:
    """Turn ALL CAPS-ish tokens into nicer casing for PDF display (keeps common acronyms)."""
    if v is None:
        return "—"
    s = str(v).strip()
    if not s:
        return "—"

    # Normalize separators for nicer wrapping.
    s = s.replace("_", " ").replace(" - ", " – ").replace("-", "–")
    parts = [p.strip() for p in s.split("–")]

    keep_upper = {"PVC", "HPL", "OSB", "BSH", "KVH", "CLT", "AI"}

    def nice_word(w: str) -> str:
        if not w:
            return w
        if w.upper() in keep_upper:
            return w.upper()
        # If it's all-caps (or mostly), title-case it.
        if w.isupper():
            return w.capitalize()
        return w

    out_parts = []
    for p in parts:
        words = [nice_word(w) for w in p.split()]
        out_parts.append(" ".join(words))

    return " – ".join(out_parts)

def P(text, style_name="Cell"):
    return Paragraph((str(text) or "").replace("\n", "<br/>"), _styles()[style_name])

def _money(x):
    try:
        v = float(x)
        s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        # NBSP so "€" never goes to a new line
        return f"{s}\u00a0€"
    except: return "—"

def _fmt_m2(v):
    try:
        val = float(v)
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        # NBSP so "m²" never goes to a new line
        return f"{s}\u00a0m²"
    except: return "—"

def _fmt_qty(v, unit=""):
    try:
        val = float(v)
        s = f"{val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        u = (unit or "").strip()
        ul = u.lower()
        if ul in {"buc", "buc.", "bucata", "bucată", "piece", "pieces", "stk", "stk.", "stück", "stücke"}:
            u = "Stk."
        if ul in {"ml", "ml.", "lfm"}:
            u = "m"
        # NBSP so the unit never goes to a new line
        return f"{s}\u00a0{u}".strip()
    except: return "—"

def _auto_col_widths(rows: list[list[str]], total_width_pt: float, font_name: str, font_size: float, pad_pt: float = 14.0, min_col_pt: float = 30.0) -> list[float]:
    """Auto-fit table columns to content, keeping within total_width_pt."""
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    maxw = [0.0] * ncols
    for r in rows:
        for i in range(ncols):
            txt = str(r[i]) if i < len(r) else ""
            w = stringWidth(txt, font_name, font_size) + pad_pt
            if w > maxw[i]:
                maxw[i] = w
    # Add a bit of breathing room so columns don't feel cramped
    widths = [max(min_col_pt, w + 4.0) for w in maxw]
    total = sum(widths)
    if total <= total_width_pt:
        return widths
    # scale down but keep min_col_pt
    flex = [max(0.0, w - min_col_pt) for w in widths]
    flex_total = sum(flex)
    if flex_total <= 0:
        return [min_col_pt] * ncols
    over = total - total_width_pt
    ratio = over / flex_total
    return [w - max(0.0, w - min_col_pt) * ratio for w in widths]

# ---------- PDF DRAWING ----------
def _draw_ribbon(canv: Canvas):
    canv.saveState()
    x, y = 18*mm, A4[1]-23*mm
    w, h = A4[0]-36*mm, 9*mm
    canv.setFillColor(colors.HexColor("#1c1c1c"))
    canv.rect(x, y, w, h, stroke=0, fill=1)
    canv.setFillColor(colors.white)
    canv.setFont(BOLD_FONT, 10)
    canv.drawString(x+6*mm, y+2.35*mm, "ANGEBOT – UNVERBINDLICHE KOSTENSCHÄTZUNG (RICHTWERT) ±10 %")
    canv.restoreState()

def _draw_footer(canv: Canvas):
    canv.saveState()
    y = 9*mm
    colw = (A4[0]-36*mm)/3.0
    x0 = 18*mm
    canv.setFont(BASE_FONT, 6.6)
    canv.setFillColor(colors.black)
    for i, block in enumerate((COMPANY["footer_left"], COMPANY["footer_mid"], COMPANY["footer_right"])):
        tx = x0 + i*colw
        lines = block.split("\n")
        for idx, line in enumerate(lines):
            canv.drawString(tx, y + (len(lines)-idx-1)*3.0*mm, line)
    canv.restoreState()

def _draw_firstpage_right_box(canv: Canvas, offer_no: str, handler: str):
    canv.saveState()
    box_x = A4[0]-18*mm-65*mm
    box_y = A4[1]-62*mm
    cw = 65*mm
    row_h = 8.2*mm
    rows = [("Datum", datetime.now().strftime("%d.%m.%Y")), ("Bearbeiter", handler), ("Kunden-Nr.", "—"), ("Auftrag", offer_no)]
    canv.setFont(BASE_FONT, 9)
    canv.setStrokeColor(colors.black)
    canv.rect(box_x, box_y - row_h*len(rows), cw, row_h*len(rows), stroke=1, fill=0)
    for i, (k, v) in enumerate(rows):
        y = box_y - (i+1)*row_h + 2.6*mm
        canv.drawString(box_x+3*mm, y, k)
        canv.drawRightString(box_x+cw-3*mm, y, v)
    canv.restoreState()

def _first_page_canvas(offer_no: str, handler: str, assets: dict | None = None):
    def _inner(canv: Canvas, doc):
        _draw_ribbon(canv)
        a = assets or {}
        identity_file = a.get("identity_image")
        show_logos = a.get("show_offer_logos", True)
        logos_file = a.get("offer_logos_image")

        identity_path = _asset_path(identity_file) if identity_file else IMG_IDENTITY
        if identity_path and identity_path.exists():
            canv.drawImage(str(identity_path), A4[0]-18*mm-85*mm, A4[1]-53*mm, 85*mm, 22*mm, preserveAspectRatio=True, mask='auto')

        if show_logos:
            logos_path = _asset_path(logos_file) if logos_file else IMG_LOGOS
            if logos_path and logos_path.exists():
                canv.drawImage(str(logos_path), 18*mm, A4[1]-55*mm, 80*mm, 26*mm, preserveAspectRatio=True, mask='auto', anchor='sw')
        _draw_firstpage_right_box(canv, offer_no, handler)
        _draw_footer(canv)
    return _inner

def _later_pages_canvas(canv: Canvas, doc):
    _draw_ribbon(canv)
    _draw_footer(canv)

def _header_block(story, styles, offer_no: str, client: dict, enforcer, assets: dict | None = None, tenant_slug: str = None):
    left_lines = [COMPANY["legal"], *COMPANY["addr_lines"], "", f"Tel. {COMPANY['phone']}", f"Fax {COMPANY['fax']}", "", COMPANY["email"], COMPANY["web"]]
    tbl = Table([[P("<br/>".join(left_lines), "Small"), P("", "Small")]], colWidths=[95*mm, A4[0]-36*mm-95*mm])
    tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP")]))
    
    # Verifică dacă există iconițe (identity_image sau offer_logos)
    a = assets or {}
    identity_file = a.get("identity_image")
    show_logos = a.get("show_offer_logos", True)
    logos_file = a.get("offer_logos_image")
    
    has_identity = False
    has_logos = False
    
    if identity_file:
        identity_path = _asset_path(identity_file)
        has_identity = identity_path.exists() if identity_path else False
    else:
        has_identity = IMG_IDENTITY.exists()
    
    if show_logos:
        if logos_file:
            logos_path = _asset_path(logos_file)
            has_logos = logos_path.exists() if logos_path else False
        else:
            has_logos = IMG_LOGOS.exists()
    
    # Pentru holzbau@holzbot.com, mutăm textul mai sus (nu au iconițe)
    # Dacă nu există iconițe SAU dacă este tenant holzbau, mutăm textul mai sus
    is_holzbau = tenant_slug and tenant_slug.lower() == "holzbau"
    if not has_identity and not has_logos or is_holzbau:
        story.append(Spacer(1, 5*mm))  # Mutat și mai sus pentru holzbau
    else:
        story.append(Spacer(1, 36*mm))
    
    story.append(tbl)
    story.append(Spacer(1, 6*mm))
    
    story.append(Paragraph(f"{enforcer.get('Angebot')} • {enforcer.get('Nr.')}: {offer_no}", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    # Datele clientului - verificăm mai multe surse și adăugăm debug
    name = client.get("nume") or client.get("name") or "—"
    city = client.get("localitate") or client.get("city") or "—"
    telefon = client.get("telefon") or "—"
    email = client.get("email") or "—"
    
    print(f"🔍 [PDF] Client data: name='{name}', city='{city}', telefon='{telefon}', email='{email}'")
    print(f"🔍 [PDF] Client dict keys: {list(client.keys()) if isinstance(client, dict) else 'Not a dict'}")
    
    name = enforcer.get(name.strip()) if name and name != "—" else "—"
    city = enforcer.get(city.strip()) if city and city != "—" else "—"
    
    lines = [
        f"<b>{enforcer.get('Bauherr / Kunde')}:</b> {name}", 
        f"<b>{enforcer.get('Ort / Bauort')}:</b> {city}", 
        f"<b>{enforcer.get('Telefon')}:</b> {telefon}", 
        f"<b>{enforcer.get('E-Mail')}:</b> {email}"
    ]
    story.append(Paragraph("<br/>".join(lines), _styles()["Cell"]))
    story.append(Spacer(1, 6*mm))

def _intro(story, styles, client: dict, enforcer: GermanEnforcer, offer_title: str | None = None):
    title = offer_title or enforcer.get("Angebot für Ihr Chiemgauer Massivholzhaus")
    # Elimină "Holzbau" din titlu dacă există
    if title and "Holzbau" in title:
        title = title.replace("Holzbau ", "").replace("Holzbau", "")
    story.append(Paragraph(title, styles["H2"]))
    story.append(Paragraph(enforcer.get("Sehr geehrte Damen und Herren,"), styles["Body"]))
    
    story.append(Paragraph(
        enforcer.get("Vielen Dank für Ihre Anfrage. Nachfolgend erhalten Sie unsere detaillierte Kostenschätzung, basierend auf den übermittelten Planunterlagen."), 
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Diese Dokumentation soll Ihnen eine klare Übersicht der notwendigen Investition bieten. Sollten Sie Fragen zur Kalkulation, den Komponenten oder wünschen Sie eine individuelle Anpassung, stehen wir Ihnen jederzeit gerne zur Verfügung."), 
        styles["Body"]
    ))
    
    story.append(Spacer(1, 6*mm))

# ---------- TABLES (TRADUCERE DIRECTĂ) ----------

def _table_standard(story, styles, title: str, data_dict: dict, enforcer: GermanEnforcer, show_mod_column: bool | None = None):
    items = data_dict.get("items", []) or data_dict.get("detailed_items", [])
    if not items: 
        return

    def _split_label_and_material(raw: str) -> tuple[str, str]:
        """Split 'Bauteil (Material)' into ('Bauteil', 'Material')."""
        if not raw or not isinstance(raw, str):
            return ("—", "—")
        mats = re.findall(r"\(([^)]+)\)", raw)
        if mats:
            base = re.sub(r"\s*\([^)]*\)\s*", " ", raw).strip()
            # also remove stray parens (some inputs are messy)
            base = re.sub(r"[()]+", " ", base).strip()
            return (base if base else raw, mats[-1].strip() if mats[-1].strip() else "—")
        # remove stray parens even if we don't have a match
        raw_clean = re.sub(r"[()]+", " ", raw).strip()
        return (raw_clean if raw_clean else raw, "—")

    # Detect if we should render Material / Mod columns for this table.
    has_any_material = False
    has_any_mode = False
    for it in items:
        name_raw = it.get("name", "")
        _label, mat_from_name = _split_label_and_material(name_raw)
        mat = it.get("material") or mat_from_name
        mode = it.get("construction_mode") or it.get("mode")
        if mat and mat != "—":
            has_any_material = True
        if mode:
            has_any_mode = True
    show_material = has_any_material
    # If show_mod_column is explicitly set, use it; otherwise auto-detect
    show_mode = has_any_mode if show_mod_column is None else show_mod_column
    
    # ✅ COLECTEAZĂ tot textul
    all_texts = [title, "Bauteil", "Fläche", "Preis/m²", "Gesamt", "SUMME"]
    if show_material:
        all_texts.append("Material")
    if show_mode:
        all_texts.append("Mod")
    for it in items:
        name_raw = it.get("name", "")
        name_clean, mat_from_name = _split_label_and_material(name_raw)
        all_texts.append(name_clean)
        if show_material:
            all_texts.append(_pretty_label(str(it.get("material") or mat_from_name or "—")))
        if show_mode:
            all_texts.append(_pretty_label(str(it.get("construction_mode") or it.get("mode") or "—")))
    
    # ✅ TRADUCE în batch
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE tabelul
    title_para = Paragraph(translations[title], styles["H2"])
    
    head = [P(translations["Bauteil"], "CellBold")]
    if show_material:
        head.append(P(translations.get("Material", "Material"), "CellBold"))
    if show_mode:
        head.append(P(translations.get("Mod", "Mod"), "CellBold"))
    head.extend([
        P(translations["Fläche"], "CellBold"), 
        P(translations["Preis/m²"], "CellBold"), 
        P(translations["Gesamt"], "CellBold"),
    ])
    
    data = []
    floor_header_rows = []  # Track which rows are floor headers
    
    # Pentru finisaje, grupăm pe etaje
    is_finishes = "Oberflächen" in title or "Ausbau" in title or "Finisaje" in title or any(it.get("category", "").startswith("finish") for it in items)
    
    if is_finishes:
        # Grupăm items după floor_label
        items_by_floor = {}
        items_without_floor = []
        
        for it in items:
            floor_label = it.get("floor_label", "")
            if floor_label:
                if floor_label not in items_by_floor:
                    items_by_floor[floor_label] = []
                items_by_floor[floor_label].append(it)
            else:
                items_without_floor.append(it)
        
        # Sortăm etajele
        floor_order = ["Erdgeschoss"]
        for i in range(1, 10):
            floor_order.append(f"Obergeschoss {i}")
        floor_order.extend(["Mansardă", "Dachgeschoss"])
        
        # Adăugăm items grupate pe etaje
        for floor_label in floor_order:
            if floor_label in items_by_floor:
                # Adăugăm header pentru etaj (dacă avem mai multe etaje)
                if len(items_by_floor) > 1 or items_without_floor:
                    floor_row = [P(f"<b>{floor_label}</b>", "CellBold")]
                    if show_material:
                        floor_row.append(P(""))
                    if show_mode:
                        floor_row.append(P(""))
                    floor_row.extend([P(""), P(""), P("")])
                    data.append(floor_row)
                    floor_header_rows.append(len(data) - 1)  # Track this row index (0-based in data, +1 for head)
                
                for it in items_by_floor[floor_label]:
                    name_original = it.get("name", "—")
                    # Eliminăm floor_label din nume dacă este deja inclus
                    if f" - {floor_label}" in name_original:
                        name_original = name_original.replace(f" - {floor_label}", "")
                    
                    name_clean, mat_from_name = _split_label_and_material(name_original)
                    name_translated = translations.get(name_clean, name_clean)

                    row = [P(name_translated)]
                    if show_material:
                        mat_val_raw = str(it.get("material") or mat_from_name or "—")
                        mat_val = _pretty_label(mat_val_raw)
                        row.append(P(translations.get(mat_val, mat_val), "CellSmall"))
                    if show_mode:
                        mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
                        mode_val = _pretty_label(mode_val_raw)
                        row.append(P(translations.get(mode_val, mode_val), "CellSmall"))

                    row.extend([
                        P(_fmt_m2(it.get("area_m2", 0))), 
                        P(_money(it.get("unit_price", 0)), "CellSmall"), 
                        P(_money(it.get("cost", 0) or it.get("total_cost", 0)), "CellBold"),
                    ])
                    data.append(row)
        
        # Adăugăm items fără floor_label (fallback pentru compatibilitate)
        for it in items_without_floor:
            name_original = it.get("name", "—")
            name_clean, mat_from_name = _split_label_and_material(name_original)
            name_translated = translations.get(name_clean, name_clean)

            row = [P(name_translated)]
            if show_material:
                mat_val_raw = str(it.get("material") or mat_from_name or "—")
                mat_val = _pretty_label(mat_val_raw)
                row.append(P(translations.get(mat_val, mat_val), "CellSmall"))
            if show_mode:
                mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
                mode_val = _pretty_label(mode_val_raw)
                row.append(P(translations.get(mode_val, mode_val), "CellSmall"))

            row.extend([
                P(_fmt_m2(it.get("area_m2", 0))), 
                P(_money(it.get("unit_price", 0)), "CellSmall"), 
                P(_money(it.get("cost", 0) or it.get("total_cost", 0)), "CellBold"),
            ])
            data.append(row)
    else:
        # Comportament normal pentru alte categorii
        for it in items:
            name_original = it.get("name", "—")
            name_clean, mat_from_name = _split_label_and_material(name_original)
            name_translated = translations.get(name_clean, name_clean)

            row = [P(name_translated)]
            if show_material:
                mat_val_raw = str(it.get("material") or mat_from_name or "—")
                mat_val = _pretty_label(mat_val_raw)
                row.append(P(translations.get(mat_val, mat_val), "CellSmall"))
            if show_mode:
                mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
                mode_val = _pretty_label(mode_val_raw)
                row.append(P(translations.get(mode_val, mode_val), "CellSmall"))

            row.extend([
                P(_fmt_m2(it.get("area_m2", 0))), 
                P(_money(it.get("unit_price", 0)), "CellSmall"), 
                P(_money(it.get("cost", 0)), "CellBold"),
            ])
            data.append(row)
    
    total_row = [P(translations["SUMME"], "CellBold")]
    if show_material:
        total_row.append("")
    if show_mode:
        total_row.append("")
    total_row.extend(["", "", P(_money(data_dict.get("total_cost", 0)), "CellBold")])
    data.append(total_row)
    
    # Auto-fit column widths tightly to text (keeps numeric cols narrow)
    head_txt = ["Bauteil"]
    if show_material: head_txt.append("Material")
    if show_mode: head_txt.append("Mod")
    head_txt += ["Fläche", "Preis/m²", "Gesamt"]

    rows_txt = [head_txt]
    for it in items:
        name_original = it.get("name", "—")
        name_clean, mat_from_name = _split_label_and_material(name_original)
        row_txt = [translations.get(name_clean, name_clean)]
        if show_material:
            mat_val_raw = str(it.get("material") or mat_from_name or "—")
            row_txt.append(translations.get(_pretty_label(mat_val_raw), _pretty_label(mat_val_raw)))
        if show_mode:
            mode_val_raw = str(it.get("construction_mode") or it.get("mode") or "—")
            row_txt.append(translations.get(_pretty_label(mode_val_raw), _pretty_label(mode_val_raw)))
        row_txt += [_fmt_m2(it.get("area_m2", 0)), _money(it.get("unit_price", 0)), _money(it.get("cost", 0))]
        rows_txt.append(row_txt)
    rows_txt.append([translations.get("SUMME", "SUMME")] + ([""] * (len(head_txt) - 2)) + [_money(data_dict.get("total_cost", 0))])

    usable_width = A4[0] - 36*mm
    col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=10, min_col_pt=26)
    align_from_col = len(head_txt) - 3

    # Improve width balance: cap numeric columns and let Bauteil take the remaining space.
    usable_width = A4[0] - 36*mm
    ncols = len(head_txt)
    # indices from end: Fläche, Preis/m², Gesamt
    idx_flaeche = ncols - 3
    idx_preis = ncols - 2
    idx_gesamt = ncols - 1
    caps = [None] * ncols
    caps[idx_flaeche] = 92.0
    caps[idx_preis] = 92.0
    caps[idx_gesamt] = 98.0
    # cap Material/Mod a bit too
    if show_material:
        caps[1] = 120.0
    if show_mode:
        caps[2 if show_material else 1] = 90.0

    capped = []
    for i, w in enumerate(col_widths):
        cw = min(w, caps[i]) if caps[i] is not None else w
        capped.append(cw)
    # give remaining space to first column (Bauteil)
    rest = usable_width - sum(capped[1:])
    capped[0] = max(120.0, rest)
    col_widths = capped

    tbl = Table([head] + data, colWidths=col_widths)
    
    # Construim stilurile - folosim culori din paleta (coffee/sand/caramel)
    # Header: sand (#F1E6D3) - mai deschis
    # Alternating rows: coffee-800 (#3E2C22) cu opacitate redusă sau sand cu opacitate
    style_commands = [
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (align_from_col,1), (-1,-1), "RIGHT")
    ]
    
    # Pentru finisaje, adăugăm background pentru header-urile de etaj - folosim caramel (#D8A25E) cu opacitate
    if is_finishes and floor_header_rows:
        for row_idx in floor_header_rows:
            # row_idx este index în data, trebuie +1 pentru că head este la 0
            style_commands.append(("BACKGROUND", (0, row_idx + 1), (-1, row_idx + 1), colors.HexColor("#D8A25E")))  # caramel pentru floor headers
    
    tbl.setStyle(TableStyle(style_commands))
    
    # Keep title + table together (no title on one page and table on next)
    story.append(KeepTogether([title_para, tbl, Spacer(1, 4*mm)]))
    story.append(Spacer(1, 2*mm))

def _table_roof_quantities(story, styles, pricing_data: dict, enforcer: GermanEnforcer):
    roof = pricing_data.get("breakdown", {}).get("roof", {})
    items = roof.get("items", []) or roof.get("detailed_items", [])
    display_items = [it for it in items if "extra_walls" not in str(it.get("category", ""))]
    
    if not display_items: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = ["Dachkonstruktion – Detail", "Komponente", "Material", "Bemerkung", "Menge", "Preis", "SUMME DACH"]
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        all_texts.append(name_clean)
        det = it.get("details", "") or ""
        mats = re.findall(r"\(([^)]+)\)", det) if isinstance(det, str) else []
        cat = str(it.get("category", "")).lower()
        is_cover = ("roof_cover" in cat) or ("eindeck" in str(it.get("name","")).lower())
        # Material rules:
        # - Dachstruktur / Spenglerarbeiten / Dämmung: blank in Material column
        # - Dacheindeckung: show selected material
        mat = (it.get("material") or (mats[-1].strip() if mats else "—")) if is_cover else ""
        # Dacheindeckung must have a remark
        if is_cover and (not det or not str(det).strip()):
            det = "Eindeckungsmaterial"
        if isinstance(det, str) and det:
            det = re.sub(r"\s*Material[^()]*\([^)]*\)\s*", " ", det, flags=re.IGNORECASE).strip()
        all_texts.append(_pretty_label(str(mat)) if mat else "")
        all_texts.append(det)
        unit_raw = str(it.get("unit", "") or "")
        all_texts.append(unit_raw)
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE
    title_para = Paragraph(translations["Dachkonstruktion – Detail"], styles["H2"])
    
    head = [
        P(translations["Komponente"], "CellBold"), 
        P(translations.get("Material", "Material"), "CellBold"),
        P(translations["Bemerkung"], "CellBold"), 
        P(translations["Menge"], "CellBold"), 
        P(translations["Preis"], "CellBold")
    ]
    
    data = []
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        det = it.get("details", "") or ""
        mats = re.findall(r"\(([^)]+)\)", det) if isinstance(det, str) else []
        cat = str(it.get("category", "")).lower()
        is_cover = ("roof_cover" in cat) or ("eindeck" in str(it.get("name","")).lower())
        mat = (it.get("material") or (mats[-1].strip() if mats else "—")) if is_cover else ""
        if is_cover and (not det or not str(det).strip()):
            det = "Eindeckungsmaterial"
        if isinstance(det, str) and det:
            det = re.sub(r"\s*Material[^()]*\([^)]*\)\s*", " ", det, flags=re.IGNORECASE).strip()
        
        unit_raw = str(it.get("unit", "") or "")
        
        data.append([
            P(translations.get(name_clean, name_clean)), 
            P(translations.get(_pretty_label(str(mat)), _pretty_label(str(mat))), "CellSmall") if mat else P("", "CellSmall"),
            P(translations.get(det, det), "CellSmall"), 
            P(_fmt_qty(it.get("quantity", 0), translations.get(unit_raw, unit_raw)), "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME DACH"], "CellBold"), 
        "", 
        "", 
        "",
        P(_money(sum(it.get("cost", 0) for it in display_items)), "CellBold"),
    ])
    
    # Auto-fit widths for this roof table
    usable_width = A4[0] - 36*mm
    roof_rows_txt = [[
        translations.get("Komponente","Komponente"),
        translations.get("Material","Material"),
        translations.get("Bemerkung","Bemerkung"),
        translations.get("Menge","Menge"),
        translations.get("Preis","Preis"),
    ]]
    for it in display_items:
        name_raw = it.get("name", "")
        name_clean = re.sub(r'\s*\([^)]*\)\s*', '', name_raw).strip()
        det = it.get("details", "") or ""
        mats = re.findall(r"\(([^)]+)\)", det) if isinstance(det, str) else []
        cat = str(it.get("category", "")).lower()
        is_cover = ("roof_cover" in cat) or ("eindeck" in str(it.get("name","")).lower())
        mat = (it.get("material") or (mats[-1].strip() if mats else "—")) if is_cover else ""
        if is_cover and (not det or not str(det).strip()):
            det = "Eindeckungsmaterial"
        if isinstance(det, str) and det:
            det = re.sub(r"\s*Material[^()]*\([^)]*\)\s*", " ", det, flags=re.IGNORECASE).strip()
        unit_u = str(it.get("unit","") or "")
        roof_rows_txt.append([
            translations.get(name_clean, name_clean),
            _pretty_label(str(mat)) if mat else "",
            translations.get(det, det),
            _fmt_qty(it.get("quantity", 0), unit_u),
            _money(it.get("cost", 0)),
        ])
    roof_rows_txt.append([translations.get("SUMME DACH","SUMME DACH"), "", "", "", _money(sum(it.get("cost", 0) for it in display_items))])
    col_widths = _auto_col_widths(roof_rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=14, min_col_pt=30)
    # Balance: keep Menge/Preis relatively narrow, give Bemerkung space
    # columns: Komponente, Material, Bemerkung, Menge, Preis
    col_widths[3] = min(col_widths[3], 95.0)   # Menge
    col_widths[4] = min(col_widths[4], 105.0)  # Preis
    # recompute Komponente to fill remaining
    rest = usable_width - (col_widths[1] + col_widths[2] + col_widths[3] + col_widths[4])
    col_widths[0] = max(110.0, rest)

    tbl = Table([head] + data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (3,1), (4,-1), "RIGHT")
    ]))
    
    story.append(KeepTogether([title_para, tbl, Spacer(1, 4*mm)]))
    story.append(Spacer(1, 2*mm))

def _table_stairs(story, styles, stairs_dict: dict, enforcer: GermanEnforcer):
    items = stairs_dict.get("detailed_items", [])
    if not items: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = ["Treppenanlagen", "Bauteil", "Beschreibung", "Menge", "Gesamt", "SUMME TREPPEN"]
    for it in items:
        all_texts.append(it.get("name", ""))
        all_texts.append(it.get("details", ""))
        all_texts.append(it.get("unit", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    # ✅ CONSTRUIEȘTE
    title_para = Paragraph(translations["Treppenanlagen"], styles["H2"])
    
    head = [
        P(translations["Bauteil"], "CellBold"), 
        P(translations["Beschreibung"], "CellBold"), 
        P(translations["Menge"], "CellBold"), 
        P(translations["Gesamt"], "CellBold")
    ]
    
    data = []
    for it in items:
        unit_raw = str(it.get("unit", "Stk.") or "Stk.")
        # normalize unit via _fmt_qty (handles buc->Stk. and prevents wrapping)
        qty_text = _fmt_qty(it.get("quantity", 0), translations.get(unit_raw, unit_raw))
        
        data.append([
            P(translations.get(it.get("name", "—"), it.get("name", "—"))), 
            P(translations.get(it.get("details", "—"), it.get("details", "—")), "CellSmall"), 
            P(qty_text, "CellSmall"), 
            P(_money(it.get("cost", 0)), "CellBold")
        ])
    
    data.append([
        P(translations["SUMME TREPPEN"], "CellBold"), 
        "", 
        "", 
        P(_money(stairs_dict.get("total_cost", 0)), "CellBold")
    ])
    
    # Auto-fit widths
    usable_width = A4[0] - 36*mm
    rows_txt = [[translations["Bauteil"], translations["Beschreibung"], translations["Menge"], translations["Gesamt"]]]
    for it in items:
        unit_raw = str(it.get("unit", "Stk.") or "Stk.")
        qty_text = _fmt_qty(it.get("quantity", 0), translations.get(unit_raw, unit_raw))
        rows_txt.append([
            translations.get(it.get("name", "—"), it.get("name", "—")),
            translations.get(it.get("details", "—"), it.get("details", "—")),
            qty_text,
            _money(it.get("cost", 0)),
        ])
    rows_txt.append([translations["SUMME TREPPEN"], "", "", _money(stairs_dict.get("total_cost", 0))])
    col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.0, pad_pt=14, min_col_pt=30)

    tbl = Table([head] + data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"), 
        ("ALIGN", (2,1), (3,-1), "RIGHT")
    ]))
    
    # Keep title + table together
    story.append(KeepTogether([title_para, tbl, Spacer(1, 4*mm)]))
    story.append(Spacer(1, 4*mm))

def _table_global_openings(story, styles, all_openings: list, enforcer: GermanEnforcer):
    if not all_openings: 
        return
    
    story.append(PageBreak())
    
    # ✅ COLECTEAZĂ texte pentru clasificare
    all_texts = [
        "Zusammenfassung Fenster & Türen", 
        "Kategorie", 
        "Stückzahl", 
        "Ø Preis/Stk.", 
        "Gesamt",
        "SUMME ÖFFNUNGEN",
        "Außentür",
        "Innentür",
        "Fenster",
        "Zweiflügelig",
        "Einflügelig"
    ]
    
    # Adaugă și textele din items pentru clasificare
    for it in all_openings:
        all_texts.append(it.get("name", ""))
        all_texts.append(it.get("type", ""))
        all_texts.append(it.get("category", ""))
        all_texts.append(it.get("location", ""))
        all_texts.append(it.get("details", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    title_para = Paragraph(translations["Zusammenfassung Fenster & Türen"], styles["H1"])
    
    groups = {}
    
    kw_window = ["fenster", "window", "glass", "fereastra"]
    kw_door = ["tür", "door", "usa", "ușă"]
    kw_ext = ["aussen", "außen", "exterior", "entrance", "haustür", "main"]
    kw_double = ["doppel", "double", "dublu", "zwei", "2-"]
    
    for it in all_openings:
        full_text = (
            str(it.get("name", "")) + " " + 
            str(it.get("type", "")) + " " + 
            str(it.get("category", "")) + " " + 
            str(it.get("location", "")) + " " +
            str(it.get("details", ""))
        ).lower()
        
        cost = float(it.get("total_cost", 0))
        
        is_window = any(x in full_text for x in kw_window)
        is_door = any(x in full_text for x in kw_door)
        
        if not (is_window or is_door): 
            continue
            
        is_ext = any(x in full_text for x in kw_ext)
        loc_str = "Außentür" if is_door and is_ext else "Innentür" if is_door else "Fenster"
        
        is_double = any(x in full_text for x in kw_double)
        type_str = translations["Zweiflügelig"] if is_double else translations["Einflügelig"]
        
        category_label = translations[loc_str]
        label = f"{category_label} ({type_str})"
            
        if label not in groups:
            groups[label] = {"n": 0, "eur": 0.0}
        groups[label]["n"] += 1
        groups[label]["eur"] += cost

    def avg(total, n): 
        return total / n if n > 0 else 0.0
    
    # Determine selected material (usually global for all openings in current pricing model)
    selected_material = None
    for it in all_openings:
        m = it.get("material")
        if m:
            selected_material = _pretty_label(str(m))
            break
    if not selected_material:
        selected_material = "—"
    else:
        # Translate material (e.g., "Lemn - Aluminiu" -> "Holz-Aluminium")
        selected_material = translations.get(selected_material, enforcer.get(selected_material))
    
    head = [
        P(translations["Kategorie"], "CellBold"), 
        P(enforcer.get("Material"), "CellBold"),
        P(translations["Stückzahl"], "CellBold"), 
        P(translations["Ø Preis/Stk."], "CellBold"), 
        P(translations["Gesamt"], "CellBold"),
    ]
    
    data = []
    total_eur = 0.0
    
    for label in sorted(groups.keys()):
        g = groups[label]
        count = g["n"]
        cost = g["eur"]
        total_eur += cost
        
        data.append([
            P(label),
            P(enforcer.get(selected_material)),
            P(str(count)),
            P(_money(avg(cost, count))),
            P(_money(cost)),
        ])

    if total_eur > 0:
        data.append([
            P(translations["SUMME ÖFFNUNGEN"], "CellBold"), 
            "", 
            "", 
            "",
            P(_money(total_eur), "CellBold"),
        ])
        
        # Auto-fit widths for openings summary (keep numeric cols narrow)
        usable_width = A4[0] - 36*mm
        rows_txt = [["Kategorie", "Material", "Stückzahl", "Ø Preis/Stk.", "Gesamt"]]
        for label in sorted(groups.keys()):
            g = groups[label]
            count = g["n"]
            cost = g["eur"]
            rows_txt.append([label, selected_material, str(count), _money(avg(cost, count)), _money(cost)])
        rows_txt.append(["SUMME ÖFFNUNGEN", "", "", "", _money(total_eur)])
        col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=10, min_col_pt=26)

        tbl = Table([head] + data, colWidths=col_widths)
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.black),
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header
            ("ALIGN", (2,1), (-1,-1), "RIGHT"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE")
        ]))
        
        story.append(KeepTogether([title_para, tbl, Spacer(1, 8*mm)]))

def _table_global_utilities(story, styles, all_utilities: list, enforcer: GermanEnforcer):
    if not all_utilities: 
        return
    
    # ✅ COLECTEAZĂ
    all_texts = [
        "Zusammenfassung Haustechnik & Installationen",
        "Elektroinstallation",
        "Heizungstechnik",
        "Abwasser",
        "Ventilation",
        "Gewerk / Kategorie",
        "Gesamtpreis",
        "SUMME HAUSTECHNIK"
    ]
    
    for it in all_utilities:
        all_texts.append(it.get("category", ""))
    
    # ✅ TRADUCE
    translations = enforcer.translate_table_batch(all_texts)
    
    title_para = Paragraph(translations["Zusammenfassung Haustechnik & Installationen"], styles["H1"])
    
    # Aggregate by category (engine emits: electricity/heating/ventilation/sewage)
    agg = {}
    total_util = 0.0
    
    for it in all_utilities:
        cat_original = it.get("category", "Sonstiges")
        cat_label = {
            "electricity": translations.get("Elektroinstallation", "Elektroinstallation"),
            "heating": translations.get("Heizungstechnik", "Heizungstechnik"),
            "ventilation": translations.get("Ventilation", "Ventilation"),
            "sewage": translations.get("Abwasser", "Abwasser"),
            "fireplace": translations.get("Kamin", "Kamin"),
            "chimney": translations.get("Kaminabzug", "Kaminabzug"),
        }.get(cat_original, translations.get(cat_original, cat_original))
        cat_translated = cat_label
        cost = it.get("total_cost", 0.0)
        total_util += cost
        agg[cat_translated] = agg.get(cat_translated, 0.0) + cost
    
    head = [
        P(translations["Gewerk / Kategorie"], "CellBold"), 
        P(translations["Gesamtpreis"], "CellBold")
    ]
    
    data = []
    for k, v in agg.items(): 
        data.append([P(k), P(_money(v))])
        
    data.append([
        P(translations["SUMME HAUSTECHNIK"], "CellBold"), 
        P(_money(total_util), "CellBold")
    ])
    
    # Auto-fit utilities table widths
    usable_width = A4[0] - 36*mm
    rows_txt = [[translations["Gewerk / Kategorie"], translations["Gesamtpreis"]]]
    for k, v in agg.items():
        rows_txt.append([k, _money(v)])
    rows_txt.append([translations["SUMME HAUSTECHNIK"], _money(total_util)])
    col_widths = _auto_col_widths(rows_txt, usable_width, BASE_FONT, 9.5, pad_pt=10, min_col_pt=26)

    tbl = Table([head] + data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("ALIGN", (1,1), (1,-1), "RIGHT"), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    
    story.append(KeepTogether([title_para, tbl, Spacer(1, 8*mm)]))

def _closing_blocks(story, styles, enforcer: GermanEnforcer):
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(enforcer.get("Annahmen & Vorbehalte"), styles["H2"]))
    story.append(Paragraph(enforcer.get("Die vorliegende Kalkulation basiert auf den übermittelten Planunterlagen."), styles["Body"]))

def _scope_clarification(story, styles, enforcer: GermanEnforcer):
    """Clarificare scop (setarea corectă a așteptărilor)"""
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(enforcer.get("Klärung des Zwecks (richtige Erwartungen setzen)"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        enforcer.get("Diese Schätzung dient der Orientierung und ist für die erste Diskussion mit dem Auftraggeber bestimmt."),
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Die Schätzung basiert auf den bereitgestellten Informationen und auf der automatischen Analyse der Pläne."),
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Das Dokument stellt kein verbindliches Angebot dar, sondern hilft bei:"),
        styles["Body"]
    ))
    story.append(Paragraph("• " + enforcer.get("schnelles Erhalten eines realistischen Budgetüberblicks"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Vermeidung von Zeitverlust bei Projekten, die finanziell nicht machbar sind"), styles["Body"]))
    story.append(Spacer(1, 3*mm))

def _project_overview(story, styles, frontend_data: dict, enforcer: GermanEnforcer, plans_data: list = None):
    """Prezentare generală proiect - DOAR informații comune (nu per etaj)"""
    story.append(Paragraph(enforcer.get("Projektübersicht"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    
    # Calculează suprafața totală, pereți și acoperiș
    total_area = 0.0
    total_interior_walls = 0.0
    total_exterior_walls = 0.0
    total_roof_area = 0.0
    num_floors = len(plans_data) if plans_data else 1
    
    # Calculează ferestre și uși (număr și suprafață totală)
    total_windows = 0
    total_doors = 0
    total_windows_area = 0.0
    total_doors_area = 0.0
    
    if plans_data:
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            total_area += pricing.get("total_area_m2", 0.0)
            
            # Numără ferestre și uși și calculează suprafața
            breakdown = pricing.get("breakdown", {})
            openings_bd = breakdown.get("openings", {})
            openings_items = openings_bd.get("items", []) or openings_bd.get("detailed_items", [])
            
            for op in openings_items:
                obj_type = str(op.get("type", "")).lower()
                area_m2 = float(op.get("area_m2", 0.0))
                
                if "window" in obj_type:
                    total_windows += 1
                    total_windows_area += area_m2
                elif "door" in obj_type:
                    total_doors += 1
                    total_doors_area += area_m2
            
            # Încarcă area_data pentru a obține pereți și acoperiș
            plan = entry.get("info")
            if plan:
                area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
                if area_json_path.exists():
                    try:
                        with open(area_json_path, "r", encoding="utf-8") as f:
                            area_data = json.load(f)
                        
                        # Pereți interiori și exteriori
                        walls_data = area_data.get("walls", {})
                        interior_walls = walls_data.get("interior", {}).get("net_area_m2", 0.0)
                        exterior_walls = walls_data.get("exterior", {}).get("net_area_m2", 0.0)
                        total_interior_walls += float(interior_walls) if interior_walls else 0.0
                        total_exterior_walls += float(exterior_walls) if exterior_walls else 0.0
                        
                        # Acoperiș
                        surfaces = area_data.get("surfaces", {})
                        roof_m2 = surfaces.get("roof_m2")
                        if roof_m2:
                            total_roof_area += float(roof_m2)
                    except Exception as e:
                        print(f"⚠️ [PDF] Eroare la încărcarea area_data pentru {plan.plan_id}: {e}")
    
    # Extrage date din formular (doar cele comune, nu per etaj)
    sistem_constructiv = frontend_data.get("sistemConstructiv", {})
    materiale_finisaj = frontend_data.get("materialeFinisaj", {})
    performanta = frontend_data.get("performanta", {})
    
    tip_sistem = sistem_constructiv.get("tipSistem", "HOLZRAHMEN")
    grad_prefabricare = sistem_constructiv.get("gradPrefabricare", "—")
    tip_fundatie = sistem_constructiv.get("tipFundatie", "—")
    tip_acoperis = sistem_constructiv.get("tipAcoperis", "Satteldach")
    material_acoperis = materiale_finisaj.get("materialAcoperis", "—")
    # Citim tipul de încălzire din performanta (mutat acolo) sau din incalzire (fallback)
    incalzire_data = frontend_data.get("incalzire", {})
    performanta_energetica = frontend_data.get("performantaEnergetica", {})
    incalzire = performanta.get("tipIncalzire") or performanta_energetica.get("tipIncalzire") or incalzire_data.get("tipIncalzire") or performanta.get("incalzire", "—")
    nivel_energetic = performanta.get("nivelEnergetic") or performanta_energetica.get("nivelEnergetic", "—")
    # Citim tipul de semineu din performantaEnergetica (prioritate), apoi performanta, apoi incalzire
    # Conform structurii din compute.utils.ts, tipSemineu poate fi în toate cele 3 locuri
    tip_semineu = (
        performanta_energetica.get("tipSemineu") or 
        performanta.get("tipSemineu") or 
        incalzire_data.get("tipSemineu") or 
        "Kein Kamin"
    )
    print(f"🔍 [PDF] tip_semineu sources: performantaEnergetica={performanta_energetica.get('tipSemineu')}, performanta={performanta.get('tipSemineu')}, incalzire={incalzire_data.get('tipSemineu')}")
    print(f"🔍 [PDF] Final tip_semineu: '{tip_semineu}'")
    tamplarie = materiale_finisaj.get("tamplarie", "—")
    nivel_finisare = materiale_finisaj.get("nivelOferta", "Schlüsselfertig")
    
    # Colectăm finisajele per etaj din plans_data
    finishes_per_floor = {}
    if plans_data:
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            breakdown = pricing.get("breakdown", {})
            finishes_bd = breakdown.get("finishes", {})
            if finishes_bd and finishes_bd.get("total_cost", 0) > 0:
                items = finishes_bd.get("detailed_items", []) or finishes_bd.get("items", [])
                for item in items:
                    floor_label = item.get("floor_label", "")
                    if not floor_label:
                        # Încercăm să extragem din nume
                        name = item.get("name", "")
                        if "Erdgeschoss" in name or "ground" in name.lower():
                            floor_label = "Erdgeschoss"
                        elif "Mansardă" in name or "Mansarda" in name:
                            floor_label = "Mansardă"
                        elif "Obergeschoss" in name:
                            # Extragem numărul
                            import re
                            match = re.search(r'Obergeschoss\s*(\d+)', name)
                            if match:
                                floor_label = f"Obergeschoss {match.group(1)}"
                            else:
                                floor_label = "Obergeschoss"
                        elif "Dachgeschoss" in name:
                            floor_label = "Dachgeschoss"
                        else:
                            # Fallback: folosim tipul planului
                            plan = entry.get("info")
                            if plan:
                                entry_type = entry.get("type", "").strip().lower()
                                if entry_type == "ground_floor":
                                    floor_label = "Erdgeschoss"
                                elif "top" in entry_type or "mansard" in entry_type:
                                    floor_label = "Dachgeschoss"
                                else:
                                    floor_label = "Obergeschoss"
                    
                    if floor_label not in finishes_per_floor:
                        finishes_per_floor[floor_label] = {"interior": None, "exterior": None}
                    
                    category = item.get("category", "")
                    material = item.get("material", "")
                    if "interior" in category:
                        finishes_per_floor[floor_label]["interior"] = material
                    elif "exterior" in category:
                        finishes_per_floor[floor_label]["exterior"] = material
    
    # Traduce valorile
    tip_sistem_de = enforcer.get(tip_sistem) if tip_sistem else "Holzrahmenbau"
    grad_prefabricare_de = enforcer.get(grad_prefabricare) if grad_prefabricare and grad_prefabricare != "—" else "—"
    tip_fundatie_de = enforcer.get(tip_fundatie) if tip_fundatie and tip_fundatie != "—" else "—"
    tip_acoperis_de = enforcer.get(tip_acoperis) if tip_acoperis else "Satteldach"
    material_acoperis_de = enforcer.get(material_acoperis) if material_acoperis and material_acoperis != "—" else "—"
    incalzire_de = enforcer.get(incalzire) if incalzire and incalzire != "—" else "—"
    nivel_energetic_de = enforcer.get(nivel_energetic) if nivel_energetic and nivel_energetic != "—" else "—"
    tamplarie_de = enforcer.get(tamplarie) if tamplarie else "—"
    nivel_finisare_de = enforcer.get(nivel_finisare) if nivel_finisare else "Schlüsselfertig"
    
    # DOAR informații comune (nu per etaj)
    overview_items = []
    # Suprafața brută totală a casei (suma tuturor etajelor)
    total_house_area = 0.0
    if plans_data:
        for idx, entry in enumerate(plans_data):
            pricing = entry.get("pricing", {})
            area = pricing.get("total_area_m2", 0.0)
            total_house_area += area
            print(f"🔍 [PDF] Plan {idx}: area={area} m² (total acum: {total_house_area} m²)")
    
    if total_house_area > 0:
        overview_items.append(f"<b>{enforcer.get('Hausfläche')}:</b> ca. <b>{total_house_area:.0f} m²</b>")
        print(f"✅ [PDF] Suprafața brută totală a casei afișată: {total_house_area:.0f} m²")
    else:
        print(f"⚠️ [PDF] total_house_area is 0 - nu se afișează suprafața casei")
    
    # Finisaje interioare și exterioare (în loc de pereți)
    if finishes_per_floor:
        # Sortăm etajele: Erdgeschoss primul, apoi Obergeschoss 1, 2, etc., apoi Mansardă/Dachgeschoss
        floor_order = ["Erdgeschoss"]
        for i in range(1, 10):
            floor_order.append(f"Obergeschoss {i}")
        floor_order.extend(["Mansardă", "Dachgeschoss"])
        
        for floor_label in floor_order:
            if floor_label in finishes_per_floor:
                floor_finishes = finishes_per_floor[floor_label]
                fin_int = floor_finishes.get("interior")
                fin_ext = floor_finishes.get("exterior")
                if fin_int:
                    fin_int_de = enforcer.get(fin_int) if fin_int else "—"
                    overview_items.append(f"<b>{enforcer.get('Finisaj interior')} ({floor_label}):</b> <b>{fin_int_de}</b>")
                if fin_ext:
                    fin_ext_de = enforcer.get(fin_ext) if fin_ext else "—"
                    overview_items.append(f"<b>{enforcer.get('Fațadă')} ({floor_label}):</b> <b>{fin_ext_de}</b>")
    
    # Suprafețe acoperiș (finisajele interioare și exterioare au fost eliminate conform cerințelor)
    if total_roof_area > 0:
        overview_items.append(f"<b>{enforcer.get('Dachfläche')}:</b> <b>{total_roof_area:.1f} m²</b>")
    
    # Număr de etaje (Stockwerke în loc de Ebenen)
    overview_items.append(f"<b>{enforcer.get('Anzahl der Stockwerke')}:</b> <b>{num_floors}</b>")
    
    # Informații despre sistem constructiv
    overview_items.append(f"<b>{enforcer.get('Bausystem')}:</b> <b>{tip_sistem_de}</b>")
    if grad_prefabricare_de != "—":
        overview_items.append(f"<b>{enforcer.get('Grad prefabricare')}:</b> <b>{grad_prefabricare_de}</b>")
    if tip_fundatie_de != "—":
        overview_items.append(f"<b>{enforcer.get('Tip fundație')}:</b> <b>{tip_fundatie_de}</b>")
    overview_items.append(f"<b>{enforcer.get('Dachtyp')}:</b> <b>{tip_acoperis_de}</b>")
    if material_acoperis_de != "—":
        overview_items.append(f"<b>{enforcer.get('Dachmaterial')}:</b> <b>{material_acoperis_de}</b>")
    
    # Semineu și horn - afișăm ce s-a selectat din formular
    print(f"🔍 [PDF] tip_semineu value: '{tip_semineu}', type: {type(tip_semineu)}")
    tip_semineu_str = str(tip_semineu).strip() if tip_semineu else ""
    # Afișăm semineul dacă există o valoare (chiar dacă este "Kein Kamin")
    if tip_semineu_str:
        semineu_de = enforcer.get(tip_semineu_str) if tip_semineu_str else tip_semineu_str
        overview_items.append(f"<b>{enforcer.get('Kamin')}:</b> <b>{semineu_de}</b>")
        # Horn este calculat per etaj, deci îl menționăm doar dacă există semineu (nu este "Kein Kamin")
        if tip_semineu_str != "Kein Kamin" and tip_semineu_str.lower() != "kein kamin":
            overview_items.append(f"<b>{enforcer.get('Kaminabzug')}:</b> <b>für {num_floors} Geschosse</b>")
            print(f"✅ [PDF] Semineu afișat: '{semineu_de}' cu horn pentru {num_floors} etaje")
        else:
            print(f"✅ [PDF] Semineu afișat: '{semineu_de}' (fără horn)")
    else:
        print(f"⚠️ [PDF] Semineu nu este afișat: tip_semineu este gol")
    
    # Sistem de încălzire și nivel energetic
    if incalzire_de != "—":
        overview_items.append(f"<b>{enforcer.get('Heizsystem')}:</b> <b>{incalzire_de}</b>")
    if nivel_energetic_de != "—":
        overview_items.append(f"<b>{enforcer.get('Nivel energetic')}:</b> <b>{nivel_energetic_de}</b>")
    overview_items.append(f"<b>{enforcer.get('Fertigstellungsgrad')}:</b> <b>{nivel_finisare_de}</b>")
    
    for item in overview_items:
        story.append(Paragraph(item, styles["Body"]))
    
    story.append(Spacer(1, 1*mm))
    story.append(Paragraph(f"({enforcer.get('gemäß verfügbaren Plänen und Informationen')})", styles["Small"]))
    story.append(Spacer(1, 3*mm))

def _floor_specific_info(story, styles, entry: dict, enforcer: GermanEnforcer):
    """Informații specifice pentru fiecare etaj"""
    plan = entry["info"]
    pricing = entry["pricing"]
    
    # Calculează suprafața pentru acest etaj
    floor_area = pricing.get("total_area_m2", 0.0)
    
    # Cost total pentru acest etaj
    floor_cost = pricing.get("total_cost_eur", 0.0)
    
    # Creează tabel cu informații specifice etajului (fără ferestre și uși - sunt la total)
    floor_info_items = []
    if floor_area > 0:
        floor_info_items.append(f"<b>{enforcer.get('Nutzfläche')}:</b> <b>{floor_area:.1f} m²</b>")
    if floor_cost > 0:
        floor_info_items.append(f"<b>{enforcer.get('Geschätzte Kosten')}:</b> <b>{_money(floor_cost)}</b>")
    
    if floor_info_items:
        story.append(Spacer(1, 2*mm))
        for item in floor_info_items:
            story.append(Paragraph(item, styles["Body"]))

def _simplified_cost_structure(story, styles, plans_data: list, inclusions: dict, enforcer: GermanEnforcer):
    """Structură de cost simplificată (eliminată complet)"""
    # Secțiunea Kostenstruktur a fost eliminată complet conform cerințelor
    pass
    
    # Calculează sumele totale pentru fiecare categorie
    foundation_total = 0.0
    structure_total = 0.0
    floors_total = 0.0
    roof_total = 0.0
    openings_total = 0.0
    finishes_total = 0.0
    utilities_total = 0.0
    fireplace_total = 0.0
    stairs_total = 0.0
    
    for entry in plans_data:
        pricing = entry["pricing"]
        bd = pricing.get("breakdown", {})
        
        # Folosim același logic ca în filtrarea breakdown-ului: verificăm inclusions pentru fiecare categorie
        if inclusions.get("foundation", False):
            foundation_total += bd.get("foundation", {}).get("total_cost", 0.0)
        # structure_walls este inclus în "structure" sau "structure_walls"
        if inclusions.get("structure_walls", False) or inclusions.get("structure", False):
            structure_total += bd.get("structure_walls", {}).get("total_cost", 0.0)
        # Verificăm și dacă există o categorie "structure" directă
        if inclusions.get("structure", False) and bd.get("structure"):
            structure_total += bd.get("structure", {}).get("total_cost", 0.0)
        if inclusions.get("floors_ceilings", False):
            floors_total += bd.get("floors_ceilings", {}).get("total_cost", 0.0)
            # Stairs sunt incluse în floors_ceilings dacă este inclus
            stairs_total += bd.get("stairs", {}).get("total_cost", 0.0)
        if inclusions.get("roof", False):
            roof_total += bd.get("roof", {}).get("total_cost", 0.0)
        if inclusions.get("openings", False):
            openings_total += bd.get("openings", {}).get("total_cost", 0.0)
        if inclusions.get("finishes", False) and bd.get("finishes"):
            finishes_total += bd.get("finishes", {}).get("total_cost", 0.0)
        if inclusions.get("utilities", False):
            utilities_total += bd.get("utilities", {}).get("total_cost", 0.0)
            # Fireplace este inclus în utilities dacă este inclus
            fireplace_total += bd.get("fireplace", {}).get("total_cost", 0.0)
        # Basement este inclus dacă foundation este inclus (basement face parte din fundație/structură)
        if inclusions.get("foundation", False):
            basement_total = bd.get("basement", {}).get("total_cost", 0.0)
            foundation_total += basement_total  # Adăugăm basement la foundation pentru consistență
    
    # Include și roof_total și floors_total în structure_total (pentru că sunt parte din structură)
    structure_total += roof_total + floors_total
    
    # Calculează totalul final (fără tabel, doar pentru verificare)
    total_cost = foundation_total + structure_total + openings_total + finishes_total + utilities_total + fireplace_total + stairs_total
    
    story.append(Spacer(1, 3*mm))

def _exclusions_section(story, styles, enforcer: GermanEnforcer):
    """Ce NU este inclus"""
    story.append(Paragraph(enforcer.get("Was NICHT enthalten ist (sehr wichtig)"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(enforcer.get("Nicht in dieser Schätzung enthalten:"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Grundstückskosten"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Außenanlagen (Zaun, Hof, Wege)"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Küche und Möbel"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Sonderausstattung oder individuelle Anforderungen"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Steuern, Genehmigungen, Anschlüsse"), styles["Body"]))
    story.append(Paragraph(enforcer.get("Diese werden in den nächsten Phasen besprochen."), styles["Body"]))
    story.append(Spacer(1, 3*mm))

def _precision_section(story, styles, enforcer: GermanEnforcer):
    """Precizia estimării"""
    story.append(Paragraph(enforcer.get("Genauigkeit der Schätzung (rechtliche Sicherheit)"), styles["H2"]))
    story.append(Spacer(1, 2*mm))
    story.append(Paragraph(
        enforcer.get("Die Schätzung liegt erfahrungsgemäß in einem Bereich von ±10-15 %, abhängig von den finalen Ausführungs- und Planungsdetails."),
        styles["Body"]
    ))
    story.append(Spacer(1, 3*mm))

def _legal_disclaimer(story, styles, enforcer: GermanEnforcer):
    """Legal disclaimer complet"""
    # Adaugă PageBreak doar dacă e necesar (nu automat)
    story.append(Spacer(1, 5*mm))
    story.append(Paragraph(enforcer.get("Rechtlicher Hinweis / Haftungsausschluss"), styles["H1"]))
    story.append(Spacer(1, 2*mm))
    
    story.append(Paragraph(
        enforcer.get("Dieses Dokument ist eine unverbindliche Kostenschätzung zur ersten Budgetorientierung und ersetzt kein verbindliches Angebot."),
        styles["Body"]
    ))
    story.append(Paragraph(
        enforcer.get("Die dargestellten Werte basieren auf den vom Nutzer bereitgestellten Informationen und typischen Erfahrungswerten der jeweiligen Holzbaufirma."),
        styles["Body"]
    ))
    story.append(Spacer(1, 2*mm))
    
    story.append(Paragraph(
        enforcer.get("Abweichungen durch Planänderungen, Ausführungsdetails, Grundstücksgegebenheiten, behördliche Auflagen oder individuelle Wünsche sind möglich."),
        styles["Body"]
    ))
    story.append(Spacer(1, 2*mm))
    
    story.append(Paragraph(enforcer.get("Nicht Bestandteil dieser Schätzung sind insbesondere:"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Außenanlagen (z. B. Einfriedungen, Einfahrten, Garten- und Landschaftsgestaltung)"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("statische Berechnungen"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("bauphysikalische Nachweise"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Grundstücks- und Bodenbeschaffenheit"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("Förderungen, Gebühren und Abgaben"), styles["Body"]))
    story.append(Paragraph("• " + enforcer.get("behördliche oder rechtliche Prüfungen"), styles["Body"]))
    story.append(Spacer(1, 2*mm))
    
    story.append(Paragraph(
        enforcer.get("Die endgültige Preisfestlegung erfolgt ausschließlich im Rahmen eines individuellen Angebots nach detaillierter Planung und Prüfung durch die ausführende Holzbaufirma."),
        styles["Body"]
    ))

# ---------- MAIN GENERATOR ----------
def generate_complete_offer_pdf(run_id: str, output_path: Path | None = None) -> Path:
    print(f"🚀 [PDF] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id)
    tenant_slug = frontend_data.get("tenant_slug") if isinstance(frontend_data, dict) else None

    branding = _apply_branding(tenant_slug)
    assets = (branding.get("assets") or {}) if isinstance(branding, dict) else {}
    company_overrides = (branding.get("company") or {}) if isinstance(branding, dict) else {}
    offer_prefix = (branding.get("offer_prefix") or "CHH") if isinstance(branding, dict) else "CHH"
    handler = (branding.get("handler_name") or "Florian Siemer") if isinstance(branding, dict) else "Florian Siemer"
    offer_title = branding.get("offer_title") if isinstance(branding, dict) else None

    # Apply company overrides (header/footer content)
    if isinstance(company_overrides, dict) and company_overrides:
        for k, v in company_overrides.items():
            if v is not None:
                if k in {"footer_left", "footer_mid", "footer_right"}:
                    COMPANY[k] = _clean_multiline_text(v, max_lines=6, max_line_len=72)
                elif k == "legal":
                    COMPANY[k] = _clean_multiline_text(v, max_lines=2, max_line_len=60).replace("\n", " ")
                elif k == "name":
                    COMPANY[k] = str(v).strip()
                elif k == "email":
                    COMPANY[k] = str(v).strip()
                elif k == "web":
                    COMPANY[k] = str(v).strip()
                elif k == "phone":
                    COMPANY[k] = str(v).strip()
                elif k == "fax":
                    COMPANY[k] = str(v).strip()
                elif k == "addr_lines" and isinstance(v, list):
                    COMPANY[k] = [str(x).strip() for x in v if str(x).strip()][:3]
                else:
                    COMPANY[k] = v
    client_data_untranslated = frontend_data.get("client", frontend_data)
    nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta", "Casă completă")
    inclusions = _get_offer_inclusions(nivel_oferta)
    
    try: 
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError: 
        plan_infos = []

    enriched_plans = []
    for plan in plan_infos:
        plan_name_parts = plan.plan_id.split('_')
        
        if len(plan_name_parts) >= 2 and plan_name_parts[-2] == 'cluster':
            meta_filename = f"{plan_name_parts[-2]}_{plan_name_parts[-1]}.json"
        else:
            meta_filename = f"{plan.plan_id}.json"

        meta_path = output_root / "plan_metadata" / meta_filename 
        
        floor_type = "unknown"
        
        if meta_path.exists():
            try: 
                raw_type = json.load(open(meta_path)).get("floor_classification", {}).get("floor_type", "unknown")
                floor_type = raw_type.strip().lower()
            except Exception as e: 
                floor_type = "unknown"
        
        if floor_type == "unknown":
            if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): 
                floor_type = "ground_floor"
            elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): 
                floor_type = "top_floor"
        
        sort_key = 0 if floor_type == "ground_floor" else 1
        enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": sort_key})
        
    enriched_plans.sort(key=lambda x: x["sort"])

    plans_data = []
    global_openings = []
    global_utilities = []

    for p_data in enriched_plans:
        plan = p_data["plan"]
        pricing_path = plan.stage_work_dir / "pricing_raw.json"
        
        if pricing_path.exists():
            with open(pricing_path, encoding="utf-8") as f: 
                p_json = json.load(f)
            
            breakdown = p_json.get("breakdown", {})
            filtered_breakdown = {}
            filtered_total = 0.0
            
            for category_key, category_data in breakdown.items():
                # Stairs este inclus dacă floors_ceilings este inclus
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Fireplace este inclus dacă utilities este inclus (fireplace face parte din utilities)
                if category_key == "fireplace" and inclusions.get("utilities", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Basement este inclus dacă foundation este inclus (basement face parte din fundație/structură)
                if category_key == "basement" and inclusions.get("foundation", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Verificăm dacă categoria este în inclusions
                if inclusions.get(category_key, False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
            
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            
            if inclusions.get("utilities", False): 
                util_bd = breakdown.get("utilities", {}) or {}
                global_utilities.extend(util_bd.get("items", []) or util_bd.get("detailed_items", []) or [])
            # Adăugăm și fireplace în global_utilities pentru afișare (doar dacă utilities este inclus)
            if inclusions.get("utilities", False) and breakdown.get("fireplace"):
                fireplace_bd = breakdown.get("fireplace", {})
                global_utilities.extend(fireplace_bd.get("items", []) or fireplace_bd.get("detailed_items", []) or [])
            if inclusions.get("openings", False): 
                global_openings.extend(breakdown.get("openings", {}).get("items", []))
            if inclusions.get("roof", False):
                roof_items = breakdown.get("roof", {}).get("items", [])
                extra_wall = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                if extra_wall and inclusions.get("structure_walls", False):
                    cost = extra_wall.get("cost", 0.0)
                    ws = filtered_breakdown.get("structure_walls", {})
                    target = next((it for it in ws.get("items", []) if "Außenverkleidung" in it.get("name","") or "Fassadenverkleidung" in it.get("name","") or "Außenwände" in it.get("name","") or "Exterior" in it.get("name","")), None)
                    if target: 
                        target["cost"] += cost
                        ws["total_cost"] += cost
                        filtered_breakdown.get("roof", {})["total_cost"] -= cost
                        filtered_total += cost

            plans_data.append({"info": plan, "type": p_data["floor_type"], "pricing": p_json})

    # ✅ INIȚIALIZARE ENFORCER (fără collect & process_translation_queue)
    print("🇩🇪 [PDF] Initializing GermanEnforcer (Direct Table Translation Mode)...")
    enforcer = GermanEnforcer()

    # --- BUILD PDF ---
    offer_no = f"{offer_prefix}-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
    doc = SimpleDocTemplate(
        str(output_path), 
        pagesize=A4, 
        leftMargin=18*mm, 
        rightMargin=18*mm, 
        topMargin=42*mm, 
        bottomMargin=22*mm, 
        title=f"Angebot {offer_no}", 
        author=COMPANY["name"]
    )
    
    styles = _styles()
    story = []
    
    _header_block(story, styles, offer_no, client_data_untranslated, enforcer, assets=assets, tenant_slug=tenant_slug) 
    _intro(story, styles, client_data_untranslated, enforcer, offer_title) 
    
    # Secțiunea 2: Prezentare generală proiect (doar informații comune)
    frontend_data_with_run_id = {**frontend_data, "run_id": run_id}
    _project_overview(story, styles, frontend_data_with_run_id, enforcer, plans_data)
    
    # Structură de cost simplificată (eliminată complet conform cerințelor)
    # _simplified_cost_structure(story, styles, plans_data, inclusions, enforcer)
    
    # Calculează ferestre și uși la total pentru afișare la planuri
    total_windows_global = 0
    total_doors_global = 0
    total_doors_interior = 0
    total_doors_exterior = 0
    total_windows_area_global = 0.0
    total_doors_area_global = 0.0
    
    if plans_data:
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            breakdown = pricing.get("breakdown", {})
            openings_bd = breakdown.get("openings", {})
            openings_items = openings_bd.get("items", []) or openings_bd.get("detailed_items", [])
            
            for op in openings_items:
                obj_type = str(op.get("type", "")).lower()
                area_m2 = float(op.get("area_m2", 0.0))
                
                if "window" in obj_type:
                    total_windows_global += 1
                    total_windows_area_global += area_m2
                elif "door" in obj_type:
                    total_doors_global += 1
                    total_doors_area_global += area_m2
                    # Status poate fi în status sau location (pentru compatibilitate)
                    # Status este păstrat în items din calculate_openings_details
                    status_raw = op.get("status", op.get("location", ""))
                    status = str(status_raw).lower()
                    
                    # Debug pentru a vedea ce se întâmplă
                    print(f"🔍 [PDF] Door item: type={obj_type}, status_raw={status_raw}, status={status}, keys={list(op.keys())}")
                    
                    # Verificăm dacă este exterior (status="exterior" sau location="Exterior")
                    if status == "exterior":
                        total_doors_exterior += 1
                        print(f"✅ [PDF] Exterior door counted: total_exterior={total_doors_exterior}")
                    else:
                        total_doors_interior += 1
                        print(f"✅ [PDF] Interior door counted: total_interior={total_doors_interior}")
        
        # Debug final
        print(f"🔍 [PDF] Final door counts: total={total_doors_global}, interior={total_doors_interior}, exterior={total_doors_exterior}")
    
    # Planuri (side by side, doar imagini, fără tabele detaliate)
    if plans_data:
        story.append(Spacer(1, 4*mm))
        # Titlul "Planungsebenen" a fost eliminat conform cerințelor
        
        # Adăugăm informații despre ferestre și uși conform pozei
        # Trebuie să citim datele din formular pentru Ausführung
        ferestre_usi = frontend_data.get("ferestreUsi", {})
        bodentiefe_fenster = ferestre_usi.get("bodentiefeFenster", "Nein")
        turhohe = ferestre_usi.get("turhohe", "Standard")
        
        # Mapăm valorile din formular la textul german pentru ferestre
        ausfuhrung_fenster_map = {
            "Nein": "Standard",
            "Ja – einzelne": "erhöhter Glasanteil",
            "Ja – mehrere / große Glasflächen": "erhöhter Glasanteil"
        }
        ausfuhrung_fenster = ausfuhrung_fenster_map.get(bodentiefe_fenster, "Standard")
        
        # Mapăm valorile din formular la textul german pentru uși
        ausfuhrung_tur_map = {
            "Standard (2,0 m)": "Standard",
            "Erhöht / Sondermaß (2,2+ m)": "Erhöht"
        }
        ausfuhrung_tur = ausfuhrung_tur_map.get(turhohe, "Standard")
        
        # Calculăm prețul total pentru ferestre și uși separat
        windows_total_price = 0.0
        doors_total_price = 0.0
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            breakdown = pricing.get("breakdown", {})
            openings_bd = breakdown.get("openings", {})
            openings_items = openings_bd.get("items", []) or openings_bd.get("detailed_items", [])
            
            for op in openings_items:
                obj_type = str(op.get("type", "")).lower()
                cost = float(op.get("total_cost", op.get("cost", 0.0)))
                
                if "window" in obj_type:
                    windows_total_price += cost
                elif "door" in obj_type:
                    doors_total_price += cost
        
        # Afișăm informații despre ferestre (cu KeepTogether pentru a rămâne pe aceeași pagină)
        if total_windows_global > 0:
            fenster_content = [
                Paragraph(f"<b>Fenster & Verglasung</b>", styles["H3"]),
                Paragraph(f"Anzahl Fenster (laut Plan): {total_windows_global}", styles["Body"]),
                Paragraph(f"Ausführung: {ausfuhrung_fenster}", styles["Body"])
            ]
            if windows_total_price > 0:
                fenster_content.append(Paragraph(f"Gesamtpreis Fenster & Verglasung: {_money(windows_total_price)}", styles["Body"]))
            fenster_content.append(Spacer(1, 2*mm))
            story.append(KeepTogether(fenster_content))
        
        # Afișăm informații despre uși (cu KeepTogether pentru a rămâne pe aceeași pagină)
        if total_doors_global > 0:
            turen_content = [
                Paragraph(f"<b>Türen</b>", styles["H3"]),
                Paragraph(f"Anzahl Türen (laut Plan): {total_doors_global}", styles["Body"]),
                Paragraph(f"Ausführung: {ausfuhrung_tur}", styles["Body"])
            ]
            if total_doors_interior > 0:
                turen_content.append(Paragraph(f"Innentüren: {total_doors_interior}", styles["Body"]))
            if total_doors_exterior > 0:
                turen_content.append(Paragraph(f"Außentüren / Balkontüren: {total_doors_exterior}", styles["Body"]))
            if doors_total_price > 0:
                turen_content.append(Paragraph(f"Gesamtpreis Türen: {_money(doors_total_price)}", styles["Body"]))
            turen_content.append(Spacer(1, 2*mm))
            story.append(KeepTogether(turen_content))
        
        # Pregătim imagini pentru afișare side by side
        plan_images = []
        plan_labels = []
        plan_info_texts = []
        
        for entry in plans_data:
            plan = entry["info"]
            pricing = entry["pricing"]
            
            # Titlurile de deasupra planurilor au fost eliminate conform cerințelor
            plan_labels.append("")  # Nu mai afișăm titluri
            
            # Pregătim textul cu informații specifice etajului (fără cost și fără suprafață)
            info_text = []
            # Nu mai afișăm suprafața la fiecare etaj conform cerințelor
            plan_info_texts.append("<br/>".join(info_text) if info_text else "")
            
            # Pregătim imaginea
            if plan.plan_image.exists():
                try:
                    im = PILImage.open(plan.plan_image).convert("L")
                    im = ImageEnhance.Brightness(im).enhance(0.9)
                    im = ImageOps.autocontrast(im)
                    width, height = im.size
                    aspect = width / height
                    # Lățime pentru side by side (2 coloane)
                    target_width = (A4[0]-36*mm-10*mm) / 2  # 10mm pentru spațiu între imagini
                    if aspect < 1: 
                        target_width = target_width * 0.9
                    img_byte_arr = io.BytesIO()
                    im.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    rl_img = Image(img_byte_arr)
                    rl_img._restrictSize(target_width, 100*mm)
                    plan_images.append(rl_img)
                except: 
                    plan_images.append(None)
            else:
                plan_images.append(None)
        
        # Afișăm planurile side by side (2 per rând)
        num_plans = len(plans_data)
        for i in range(0, num_plans, 2):
            row_images = []
            row_labels = []
            row_infos = []
            
            # Primul plan din rând
            if i < num_plans:
                row_labels.append(plan_labels[i])
                row_infos.append(plan_info_texts[i])
                row_images.append(plan_images[i] if plan_images[i] else "")
            
            # Al doilea plan din rând (dacă există)
            if i + 1 < num_plans:
                row_labels.append(plan_labels[i + 1])
                row_infos.append(plan_info_texts[i + 1])
                row_images.append(plan_images[i + 1] if plan_images[i + 1] else "")
            else:
                # Dacă avem doar un plan, adăugăm celula goală
                row_labels.append("")
                row_infos.append("")
                row_images.append("")
            
            # Creăm tabel cu 2 coloane pentru planuri side by side
            col_width = (A4[0]-36*mm-10*mm) / 2
            
            # Rând cu imagini (titlurile au fost eliminate conform cerințelor)
            img_row = [
                row_images[0] if row_images[0] else P("", "Body"),
                row_images[1] if row_images[1] else P("", "Body")
            ]
            
            # Nu mai afișăm label-uri și info (au fost eliminate conform cerințelor)
            table_data = [img_row]
            tbl = Table(table_data, colWidths=[col_width, col_width])
            tbl.setStyle(TableStyle([
                ("VALIGN", (0,0), (-1,-1), "TOP"),
                ("ALIGN", (0,0), (-1,-1), "CENTER"),
                ("LEFTPADDING", (0,0), (-1,-1), 0),
                ("RIGHTPADDING", (0,0), (-1,-1), 5*mm),
            ]))
            
            story.append(tbl)
            if i + 2 < num_plans:  # Dacă mai sunt planuri, adăugăm spațiu
                story.append(Spacer(1, 6*mm))

    # Nu mai afișăm tabele detaliate pentru openings și utilities - sunt în structura simplificată

    story.append(Spacer(1, 6*mm))
    story.append(Paragraph(enforcer.get("Gesamtkostenzusammenstellung"), styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    # Recalculăm filtered_total din breakdown-ul filtrat pentru a ne asigura că este corect
    # (plans_data are deja breakdown-ul filtrat și total_cost_eur recalculat)
    filtered_total = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
    
    # Verificare: dacă filtered_total este 0, încercăm să calculăm din breakdown direct
    if filtered_total == 0:
        filtered_total = 0.0
        for entry in plans_data:
            pricing = entry["pricing"]
            bd = pricing.get("breakdown", {})
            for cat_key, cat_data in bd.items():
                if isinstance(cat_data, dict):
                    filtered_total += cat_data.get("total_cost", 0.0)
    
    cost_margin_logistics = filtered_total * 0.10
    cost_margin_oversight = filtered_total * 0.10
    
    net = filtered_total + cost_margin_logistics + cost_margin_oversight
    vat = net * 0.19
    gross = net + vat
    
    head = [
        P(enforcer.get("Position"), "CellBold"), 
        P(enforcer.get("Betrag"), "CellBold")
    ]
    
    data = [
        [P(enforcer.get("Baukosten (Konstruktion, Ausbau, Technik)")), P(_money(filtered_total))],
        [P(enforcer.get("Baustelleneinrichtung, Logistik & Planung")), P(_money(cost_margin_logistics))],
        [P(enforcer.get("Bauleitung & Koordination")), P(_money(cost_margin_oversight))],
        [P(f"<b>{enforcer.get('Nettosumme (exkl. MwSt.)')}</b>"), P(_money(net), "CellBold")],
        [P(enforcer.get("MwSt. (19%)")), P(_money(vat))],
        [P(f"<b>{enforcer.get('GESAMTSUMME BRUTTO')}</b>"), P(_money(gross), "H2")],
    ]
    
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("ALIGN", (1,1), (1,-1), "RIGHT"), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    
    story.append(tbl)
    story.append(Spacer(1, 5*mm))
    
    # Legal disclaimer
    _legal_disclaimer(story, styles, enforcer)
    
    doc.build(
        story, 
        onFirstPage=_first_page_canvas(offer_no, handler, assets=assets), 
        onLaterPages=_later_pages_canvas
    )
    
    print(f"✅ [PDF] Generat Final (DE): {output_path}")
    return output_path

# ---------- ADMIN PDF GENERATOR (STRICT, NO BRANDING) ----------
def generate_admin_offer_pdf(run_id: str, output_path: Path | None = None) -> Path:
    """
    Generează un PDF strict pentru admin, fără branding fancy, fără "povesti",
    doar date brute și tabele cu prețuri.
    """
    print(f"🚀 [PDF ADMIN] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"oferta_admin_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id)
    tenant_slug = frontend_data.get("tenant_slug") if isinstance(frontend_data, dict) else None
    
    try: 
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError: 
        plan_infos = []

    enriched_plans = []
    for plan in plan_infos:
        plan_name_parts = plan.plan_id.split('_')
        
        if len(plan_name_parts) >= 2 and plan_name_parts[-2] == 'cluster':
            meta_filename = f"{plan_name_parts[-2]}_{plan_name_parts[-1]}.json"
        else:
            meta_filename = f"{plan.plan_id}.json"

        meta_path = output_root / "plan_metadata" / meta_filename 
        
        floor_type = "unknown"
        
        if meta_path.exists():
            try: 
                raw_type = json.load(open(meta_path)).get("floor_classification", {}).get("floor_type", "unknown")
                floor_type = raw_type.strip().lower()
            except Exception as e: 
                floor_type = "unknown"
        
        if floor_type == "unknown":
            if "parter" in plan.plan_id.lower() or "ground" in plan.plan_id.lower(): 
                floor_type = "ground_floor"
            elif "etaj" in plan.plan_id.lower() or "top" in plan.plan_id.lower(): 
                floor_type = "top_floor"
        
        sort_key = 0 if floor_type == "ground_floor" else 1
        enriched_plans.append({"plan": plan, "floor_type": floor_type, "sort": sort_key})
        
    enriched_plans.sort(key=lambda x: x["sort"])

    plans_data = []
    global_openings = []
    global_utilities = []

    # ADMIN: Folosim aceeași filtrare ca în user PDF pentru același preț
    nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta", "Casă completă")
    inclusions = _get_offer_inclusions(nivel_oferta)
    
    for p_data in enriched_plans:
        plan = p_data["plan"]
        pricing_path = plan.stage_work_dir / "pricing_raw.json"
        
        if pricing_path.exists():
            with open(pricing_path, encoding="utf-8") as f: 
                p_json = json.load(f)
            
            breakdown = p_json.get("breakdown", {})
            filtered_breakdown = {}
            filtered_total = 0.0
            
            for category_key, category_data in breakdown.items():
                # Stairs este inclus dacă floors_ceilings este inclus
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Fireplace este inclus dacă utilities este inclus (fireplace face parte din utilities)
                if category_key == "fireplace" and inclusions.get("utilities", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Basement este inclus dacă foundation este inclus (basement face parte din fundație/structură)
                if category_key == "basement" and inclusions.get("foundation", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Verificăm dacă categoria este în inclusions
                if inclusions.get(category_key, False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
            
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            
            if inclusions.get("utilities", False): 
                util_bd = breakdown.get("utilities", {}) or {}
                global_utilities.extend(util_bd.get("items", []) or util_bd.get("detailed_items", []) or [])
            # Adăugăm și fireplace în global_utilities pentru afișare (doar dacă utilities este inclus)
            if inclusions.get("utilities", False) and breakdown.get("fireplace"):
                fireplace_bd = breakdown.get("fireplace", {})
                global_utilities.extend(fireplace_bd.get("items", []) or fireplace_bd.get("detailed_items", []) or [])
            if inclusions.get("openings", False): 
                global_openings.extend(breakdown.get("openings", {}).get("items", []))
            if inclusions.get("roof", False):
                roof_items = breakdown.get("roof", {}).get("items", [])
                extra_wall = next((it for it in roof_items if "extra_walls" in it.get("category", "")), None)
                if extra_wall and inclusions.get("structure_walls", False):
                    cost = extra_wall.get("cost", 0.0)
                    ws = filtered_breakdown.get("structure_walls", {})
                    target = next((it for it in ws.get("items", []) if "Außenverkleidung" in it.get("name","") or "Fassadenverkleidung" in it.get("name","") or "Außenwände" in it.get("name","") or "Exterior" in it.get("name","")), None)
                    if target: 
                        target["cost"] += cost
                        ws["total_cost"] += cost
                        filtered_breakdown.get("roof", {})["total_cost"] -= cost
                        filtered_total += cost
            
            plans_data.append({"info": plan, "type": p_data["floor_type"], "pricing": p_json})

    # ADMIN: Enforcer simplu, fără branding
    enforcer = GermanEnforcer()

    # --- BUILD PDF (MINIMAL) ---
    offer_no = f"ADMIN-{datetime.now().strftime('%Y')}-{random.randint(1000,9999)}"
    doc = SimpleDocTemplate(
        str(output_path), 
        pagesize=A4, 
        leftMargin=18*mm, 
        rightMargin=18*mm, 
        topMargin=20*mm, 
        bottomMargin=15*mm, 
        title=f"Admin Offer {offer_no}", 
        author="Admin"
    )
    
    styles = _styles()
    story = []
    
    # ADMIN: Header minimal, fără branding
    client_data = frontend_data.get("client", {})
    story.append(Paragraph(f"ADMIN OFFER - {offer_no}", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    
    name = (client_data.get("nume") or client_data.get("name") or "—").strip()
    city = (client_data.get("localitate") or client_data.get("city") or "—").strip()
    
    lines = [
        f"<b>{enforcer.get('Bauherr / Kunde')}:</b> {name}", 
        f"<b>{enforcer.get('Ort / Bauort')}:</b> {city}", 
        f"<b>{enforcer.get('Telefon')}:</b> {client_data.get('telefon') or '—'}", 
        f"<b>{enforcer.get('E-Mail')}:</b> {client_data.get('email') or '—'}",
        f"<b>Mandant:</b> {tenant_slug or '—'}"
    ]
    story.append(Paragraph("<br/>".join(lines), styles["Cell"]))
    story.append(Spacer(1, 4*mm))
    
    # ADMIN: Sumar cu toate datele din formular
    story.append(Paragraph("ZUSAMMENFASSUNG DER FORMULARDATEN", styles["H2"]))
    story.append(Spacer(1, 2*mm))
    
    sistem_constructiv = frontend_data.get("sistemConstructiv", {})
    materiale_finisaj = frontend_data.get("materialeFinisaj", {})
    performanta = frontend_data.get("performanta", {})
    logistica = frontend_data.get("logistica", {})
    
    summary_items = []
    
    # Sistem constructiv
    if sistem_constructiv:
        tip_sistem = sistem_constructiv.get("tipSistem", "—")
        grad_prefabricare = sistem_constructiv.get("gradPrefabricare", "—")
        tip_fundatie = sistem_constructiv.get("tipFundatie", "—")
        tip_acoperis = sistem_constructiv.get("tipAcoperis", "—")
        
        summary_items.append(f"<b>{enforcer.get('Sistem constructiv')}:</b> {enforcer.get(tip_sistem) if tip_sistem != '—' else '—'}")
        if grad_prefabricare and grad_prefabricare != "—":
            summary_items.append(f"<b>{enforcer.get('Grad prefabricare')}:</b> {enforcer.get(grad_prefabricare)}")
        if tip_fundatie and tip_fundatie != "—":
            summary_items.append(f"<b>{enforcer.get('Tip fundație')}:</b> {enforcer.get(tip_fundatie)}")
        if tip_acoperis and tip_acoperis != "—":
            summary_items.append(f"<b>{enforcer.get('Tip acoperiș')}:</b> {enforcer.get(tip_acoperis)}")
    
    # Materiale și finisaje
    if materiale_finisaj:
        nivel_oferta = materiale_finisaj.get("nivelOferta", "—")
        tamplarie = materiale_finisaj.get("tamplarie", "—")
        material_acoperis = materiale_finisaj.get("materialAcoperis", "—")
        
        summary_items.append(f"<b>{enforcer.get('Nivel ofertă')}:</b> {enforcer.get(nivel_oferta) if nivel_oferta != '—' else '—'}")
        
        # Colectăm finisajele per etaj din plans_data
        finishes_per_floor = {}
        for entry in plans_data:
            pricing = entry.get("pricing", {})
            breakdown = pricing.get("breakdown", {})
            finishes_bd = breakdown.get("finishes", {})
            if finishes_bd and finishes_bd.get("total_cost", 0) > 0:
                items = finishes_bd.get("detailed_items", []) or finishes_bd.get("items", [])
                for item in items:
                    floor_label = item.get("floor_label", "")
                    if not floor_label:
                        # Încercăm să extragem din nume
                        name = item.get("name", "")
                        if "Erdgeschoss" in name or "ground" in name.lower():
                            floor_label = "Erdgeschoss"
                        elif "Mansardă" in name or "Mansarda" in name:
                            floor_label = "Mansardă"
                        elif "Obergeschoss" in name:
                            import re
                            match = re.search(r'Obergeschoss\s*(\d+)', name)
                            if match:
                                floor_label = f"Obergeschoss {match.group(1)}"
                            else:
                                floor_label = "Obergeschoss"
                        elif "Dachgeschoss" in name:
                            floor_label = "Dachgeschoss"
                        else:
                            entry_type = entry.get("type", "").strip().lower()
                            if entry_type == "ground_floor":
                                floor_label = "Erdgeschoss"
                            elif "top" in entry_type or "mansard" in entry_type:
                                floor_label = "Dachgeschoss"
                            else:
                                floor_label = "Obergeschoss"
                    
                    if floor_label not in finishes_per_floor:
                        finishes_per_floor[floor_label] = {"interior": None, "exterior": None}
                    
                    category = item.get("category", "")
                    material = item.get("material", "")
                    if "interior" in category:
                        finishes_per_floor[floor_label]["interior"] = material
                    elif "exterior" in category:
                        finishes_per_floor[floor_label]["exterior"] = material
        
        # Afișăm finisajele per etaj
        if finishes_per_floor:
            floor_order = ["Erdgeschoss"]
            for i in range(1, 10):
                floor_order.append(f"Obergeschoss {i}")
            floor_order.extend(["Mansardă", "Dachgeschoss"])
            
            for floor_label in floor_order:
                if floor_label in finishes_per_floor:
                    floor_finishes = finishes_per_floor[floor_label]
                    fin_int = floor_finishes.get("interior")
                    fin_ext = floor_finishes.get("exterior")
                    if fin_int:
                        summary_items.append(f"<b>{enforcer.get('Finisaj interior')} ({floor_label}):</b> {enforcer.get(fin_int)}")
                    if fin_ext:
                        summary_items.append(f"<b>{enforcer.get('Fațadă')} ({floor_label}):</b> {enforcer.get(fin_ext)}")
        
        if tamplarie and tamplarie != "—":
            summary_items.append(f"<b>{enforcer.get('Fenster & Türen')}:</b> {enforcer.get(tamplarie)}")
        if material_acoperis and material_acoperis != "—":
            summary_items.append(f"<b>{enforcer.get('Dachmaterial')}:</b> {enforcer.get(material_acoperis)}")
    
    # Performanță energetică
    if performanta:
        nivel_energetic = performanta.get("nivelEnergetic", "—")
        incalzire = performanta.get("incalzire", "—")
        ventilatie = performanta.get("ventilatie", False)
        
        if nivel_energetic and nivel_energetic != "—":
            summary_items.append(f"<b>{enforcer.get('Nivel energetic')}:</b> {enforcer.get(nivel_energetic)}")
        if incalzire and incalzire != "—":
            summary_items.append(f"<b>{enforcer.get('Heizsystem')}:</b> {enforcer.get(incalzire)}")
        if ventilatie:
            summary_items.append(f"<b>{enforcer.get('Ventilație')}:</b> {enforcer.get('Ja')}")
    
    # Logistică
    if logistica:
        acces_santier = logistica.get("accesSantier", "—")
        teren = logistica.get("teren", "—")
        utilitati = logistica.get("utilitati", False)
        
        if acces_santier and acces_santier != "—":
            summary_items.append(f"<b>Baustellenzugang:</b> {enforcer.get(acces_santier)}")
        if teren and teren != "—":
            summary_items.append(f"<b>Gelände:</b> {teren}")
        if utilitati:
            summary_items.append(f"<b>Versorgungsanschlüsse:</b> Ja")
    
    # Număr ferestre și uși
    num_windows = 0
    num_doors = 0
    for op in global_openings:
        obj_type = str(op.get("type", "")).lower()
        if "window" in obj_type:
            num_windows += 1
        elif "door" in obj_type:
            num_doors += 1
    
    if num_windows > 0:
        summary_items.append(f"<b>Anzahl Fenster (detektiert):</b> {num_windows}")
    if num_doors > 0:
        summary_items.append(f"<b>Anzahl Türen (detektiert):</b> {num_doors}")
    
    for item in summary_items:
        story.append(Paragraph(item, styles["Body"]))
    
    story.append(Spacer(1, 4*mm))

    # ADMIN: Planuri cu imagini și date detaliate
    story.append(Paragraph("DETALII PLANURI & KOSTEN", styles["H2"]))
    story.append(Spacer(1, 3*mm))
    
    for entry in plans_data:
        plan = entry["info"]
        pricing = entry["pricing"]
        
        # Nu mai adăugăm PageBreak automat, doar spacing
        if entry != plans_data[0]:
            story.append(Spacer(1, 6*mm))
        
        entry_type_clean = entry["type"].strip().lower() 
        floor_label = "Erdgeschoss" if entry_type_clean == "ground_floor" else "Obergeschoss / Dachgeschoss"
        
        story.append(Paragraph(f"Plan: {floor_label} ({plan.plan_id})", styles["H2"]))
        story.append(Spacer(1, 2*mm))
        
        # ADMIN: Adaugă imaginea planului
        if plan.plan_image.exists():
            try:
                im = PILImage.open(plan.plan_image).convert("L")
                im = ImageEnhance.Brightness(im).enhance(0.9)
                im = ImageOps.autocontrast(im)
                width, height = im.size
                aspect = width / height
                target_width = A4[0]-36*mm
                if aspect < 1: 
                    target_width = (A4[0]-36*mm) * 0.65
                img_byte_arr = io.BytesIO()
                im.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                rl_img = Image(img_byte_arr)
                rl_img._restrictSize(target_width, 75*mm)
                rl_img.hAlign = 'CENTER'
                story.append(Spacer(1, 2*mm))
                story.append(rl_img)
                story.append(Spacer(1, 3*mm))
            except Exception as e:
                print(f"⚠️ [PDF ADMIN] Nu pot încărca imaginea planului: {e}")
        
        bd = pricing.get("breakdown", {})
        
        # ADMIN: Afișăm TOATE categoriile, fără filtru
        # ADMIN: Nu afișăm coloana Mod (nu se mai folosește)
        if bd.get("foundation"): 
            _table_standard(story, styles, "Fundament / Bodenplatte", bd.get("foundation", {}), enforcer, show_mod_column=False)
        if bd.get("basement"): 
            _table_standard(story, styles, "Keller / Untergeschoss", bd.get("basement", {}), enforcer, show_mod_column=False)
        if bd.get("structure_walls"): 
            _table_standard(story, styles, "Tragwerkskonstruktion – Wände", bd.get("structure_walls", {}), enforcer, show_mod_column=False)
            
            # Adaugă tabelul cu măsurătorile pereților pentru acest plan
            try:
                area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
                if area_json_path.exists():
                    with open(area_json_path, "r", encoding="utf-8") as f:
                        area_data = json.load(f)
                    
                    from .tables import create_wall_measurements_table
                    story.append(Spacer(1, 2*mm))
                    story.append(Paragraph(f"<b>{enforcer.get('Măsurători Pereți')} - {plan.plan_id}:</b>", styles["Body"]))
                    measurements_table = create_wall_measurements_table(area_data, enforcer)
                    story.append(measurements_table)
                    story.append(Spacer(1, 3*mm))
            except Exception as e:
                print(f"⚠️ [PDF ADMIN] Could not load area_data for {plan.plan_id}: {e}")
        
        if bd.get("floors_ceilings"): 
            _table_standard(story, styles, "Geschossdecken & Balken", bd.get("floors_ceilings", {}), enforcer, show_mod_column=False)
        if bd.get("stairs"): 
            _table_stairs(story, styles, bd.get("stairs", {}), enforcer)
        if bd.get("roof"): 
            _table_roof_quantities(story, styles, pricing, enforcer)
        if bd.get("finishes"): 
            _table_standard(story, styles, "Oberflächen & Ausbau", bd.get("finishes", {}), enforcer, show_mod_column=False)

    # ADMIN: Toate deschiderile și utilitățile
    if global_openings: 
        _table_global_openings(story, styles, global_openings, enforcer)
    if global_utilities: 
        _table_global_utilities(story, styles, global_utilities, enforcer)

    # ADMIN: Summary strict
    story.append(PageBreak())
    story.append(Paragraph("Gesamtkostenzusammenstellung (ADMIN)", styles["H1"]))
    
    filtered_total = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
    
    cost_margin_logistics = filtered_total * 0.10
    cost_margin_oversight = filtered_total * 0.10
    
    net = filtered_total + cost_margin_logistics + cost_margin_oversight
    vat = net * 0.19
    gross = net + vat
    
    head = [
        P("Position", "CellBold"), 
        P("Betrag", "CellBold")
    ]
    
    data = [
        [P("Baukosten (Konstruktion, Ausbau, Technik)"), P(_money(filtered_total))],
        [P("Baustelleneinrichtung, Logistik & Planung"), P(_money(cost_margin_logistics))],
        [P("Bauleitung & Koordination"), P(_money(cost_margin_oversight))],
        [P("<b>Nettosumme (exkl. MwSt.)</b>"), P(_money(net), "CellBold")],
        [P("MwSt. (19%)"), P(_money(vat))],
        [P("<b>GESAMTSUMME BRUTTO</b>"), P(_money(gross), "H2")],
    ]
    
    tbl = Table([head] + data, colWidths=[120*mm, 50*mm])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.black), 
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#F1E6D3")),  # sand pentru header 
        ("ALIGN", (1,1), (1,-1), "RIGHT"), 
        ("VALIGN", (0,0), (-1,-1), "MIDDLE")
    ]))
    
    story.append(tbl)
    
    # ADMIN: Fără closing blocks verbose, doar o notă minimală
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("ADMIN VERSION - Raw data, no branding", styles["Small"]))
    
    # ADMIN: Fără canvas fancy, doar PDF simplu
    doc.build(story)
    
    print(f"✅ [PDF ADMIN] Generat: {output_path}")
    return output_path

# ---------- ADMIN CALCULATION METHOD PDF GENERATOR (ENGLISH) ----------
def generate_admin_calculation_method_pdf(run_id: str, output_path: Path | None = None) -> Path:
    """
    Generates a detailed calculation method PDF for admin users, explaining
    all formulas, coefficients, and numerical examples in English.
    """
    print(f"🚀 [PDF CALC METHOD] START: {run_id}")
    
    jobs_root_base = PROJECT_ROOT / "jobs"
    output_root = jobs_root_base / run_id
    
    if not output_root.exists(): 
        output_root = RUNNER_ROOT / "output" / run_id
        if not output_root.exists():
            print(f"❌ EROARE: Nu găsesc directorul de output pentru run_id='{run_id}'")
            raise FileNotFoundError(f"Output nu există: {output_root}")
        print(f"⚠️ Folosind directorul standard: {output_root.resolve()}")
    else:
        print(f"✅ Folosind JOBS_ROOT: {output_root.resolve()}")

    if output_path is None:
        final_output_root = RUNNER_ROOT / "output" / run_id
        pdf_dir = final_output_root / "offer_pdf"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output_path = pdf_dir / f"calculation_method_{run_id}.pdf"

    frontend_data = load_frontend_data_for_run(run_id)
    tenant_slug = frontend_data.get("tenant_slug") if isinstance(frontend_data, dict) else None
    
    try: 
        plan_infos = load_plan_infos(run_id, stage_name="pricing")
    except PlansListError: 
        plan_infos = []

    # Load pricing coefficients
    from pricing.db_loader import fetch_pricing_parameters
    calc_mode = frontend_data.get("calc_mode") or "default"
    try:
        pricing_coeffs = fetch_pricing_parameters(tenant_slug, calc_mode=calc_mode)
    except Exception as e:
        print(f"⚠️ Could not load pricing coefficients: {e}")
        pricing_coeffs = {}

    # Load actual pricing data for each plan - folosim aceeași filtrare ca în user PDF
    nivel_oferta = frontend_data.get("materialeFinisaj", {}).get("nivelOferta", "Casă completă")
    inclusions = _get_offer_inclusions(nivel_oferta)
    
    plans_data = []
    for plan in plan_infos:
        pricing_path = plan.stage_work_dir / "pricing_raw.json"
        if pricing_path.exists():
            with open(pricing_path, encoding="utf-8") as f: 
                p_json = json.load(f)
            
            breakdown = p_json.get("breakdown", {})
            filtered_breakdown = {}
            filtered_total = 0.0
            
            for category_key, category_data in breakdown.items():
                # Stairs este inclus dacă floors_ceilings este inclus
                if category_key == "stairs" and inclusions.get("floors_ceilings", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Fireplace este inclus dacă utilities este inclus (fireplace face parte din utilities)
                if category_key == "fireplace" and inclusions.get("utilities", False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
                    continue
                
                # Verificăm dacă categoria este în inclusions
                if inclusions.get(category_key, False):
                    filtered_breakdown[category_key] = category_data
                    filtered_total += category_data.get("total_cost", 0.0)
            
            p_json["breakdown"] = filtered_breakdown
            p_json["total_cost_eur"] = filtered_total
            
            plans_data.append({"info": plan, "pricing": p_json})

    # Build PDF
    doc = SimpleDocTemplate(
        str(output_path), 
        pagesize=A4, 
        leftMargin=18*mm, 
        rightMargin=18*mm, 
        topMargin=20*mm, 
        bottomMargin=15*mm, 
        title=f"Calculation Method - {run_id}", 
        author="Admin"
    )
    
    styles = _styles()
    story = []
    
    # Title
    story.append(Paragraph("CALCULATION METHOD DOCUMENTATION", styles["H1"]))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(f"<b>Run ID:</b> {run_id}", styles["Body"]))
    story.append(Paragraph(f"<b>Tenant:</b> {tenant_slug or '—'}", styles["Body"]))
    story.append(Spacer(1, 4*mm))
    
    # Introduction
    story.append(Paragraph("Introduction", styles["H2"]))
    story.append(Paragraph(
        "This document provides a detailed explanation of the pricing calculation method used for this specific project. "
        "For each plan, you will find the exact materials selected, the formulas used, the coefficients applied, "
        "and the complete numerical calculations with actual values.",
        styles["Body"]
    ))
    story.append(Spacer(1, 4*mm))
    
    # Extract user selections from frontend_data
    sistem_constructiv = frontend_data.get("sistemConstructiv", {})
    materiale_finisaj = frontend_data.get("materialeFinisaj", {})
    performanta = frontend_data.get("performanta", {})
    ferestre_usi = frontend_data.get("ferestreUsi", {})
    incalzire = frontend_data.get("incalzire", {})
    
    # User selections summary
    story.append(Paragraph("User Selections for This Project", styles["H2"]))
    
    # Construction system
    tip_sistem = sistem_constructiv.get("tipSistem", "—")
    grad_prefabricare = sistem_constructiv.get("gradPrefabricare", "—")
    tip_fundatie = sistem_constructiv.get("tipFundatie", "—")
    tip_acoperis = sistem_constructiv.get("tipAcoperis", "—")
    
    story.append(Paragraph(f"<b>Construction System:</b> {tip_sistem}", styles["Body"]))
    if grad_prefabricare and grad_prefabricare != "—":
        story.append(Paragraph(f"<b>Prefabrication Level:</b> {grad_prefabricare}", styles["Body"]))
    if tip_fundatie and tip_fundatie != "—":
        story.append(Paragraph(f"<b>Foundation Type:</b> {tip_fundatie}", styles["Body"]))
    if tip_acoperis and tip_acoperis != "—":
        story.append(Paragraph(f"<b>Roof Type:</b> {tip_acoperis}", styles["Body"]))
    
    # Window and door settings
    bodentiefe_fenster = ferestre_usi.get("bodentiefeFenster", "—")
    window_quality = ferestre_usi.get("windowQuality", "—")
    turhohe = ferestre_usi.get("turhohe", "—")
    
    if bodentiefe_fenster and bodentiefe_fenster != "—":
        story.append(Paragraph(f"<b>Window Height Setting:</b> {bodentiefe_fenster}", styles["Body"]))
    if window_quality and window_quality != "—":
        story.append(Paragraph(f"<b>Window Quality:</b> {window_quality}", styles["Body"]))
    if turhohe and turhohe != "—":
        story.append(Paragraph(f"<b>Door Height:</b> {turhohe}", styles["Body"]))
    
    # Energy and heating
    nivel_energetic = performanta.get("nivelEnergetic", "—")
    tip_incalzire = performanta.get("tipIncalzire", "—")
    ventilatie = performanta.get("ventilatie", False)
    semineu = incalzire.get("semineu", False)
    
    if nivel_energetic and nivel_energetic != "—":
        story.append(Paragraph(f"<b>Energy Level:</b> {nivel_energetic}", styles["Body"]))
    if tip_incalzire and tip_incalzire != "—":
        story.append(Paragraph(f"<b>Heating Type:</b> {tip_incalzire}", styles["Body"]))
    story.append(Paragraph(f"<b>Ventilation:</b> {'Yes' if ventilatie else 'No'}", styles["Body"]))
    story.append(Paragraph(f"<b>Fireplace:</b> {'Yes' if semineu else 'No'}", styles["Body"]))
    
    story.append(Spacer(1, 6*mm))
    
    # Iterate through each plan and show specific calculations
    for plan_idx, entry in enumerate(plans_data, 1):
        plan = entry["info"]
        pricing = entry["pricing"]
        breakdown = pricing.get("breakdown", {})
        
        # Plan header
        story.append(PageBreak() if plan_idx > 1 else Spacer(1, 0))
        story.append(Paragraph(f"PLAN {plan_idx}: {plan.plan_id}", styles["H1"]))
        story.append(Spacer(1, 3*mm))
        
        # 1. FOUNDATION CALCULATION FOR THIS PLAN
        foundation = breakdown.get("foundation", {})
        if foundation and foundation.get("total_cost", 0) > 0:
            story.append(Paragraph("1. Foundation Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Foundation Area (m²) × Unit Price per m²",
                styles["Body"]
            ))
            
            items = foundation.get("detailed_items", [])
            for item in items:
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                foundation_type = item.get("name", "").split("(")[1].split(")")[0] if "(" in item.get("name", "") else "—"
                
                story.append(Paragraph(
                    f"<b>Foundation Type Selected:</b> {foundation_type}",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"<b>Calculation:</b> {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 2. WALLS CALCULATION FOR THIS PLAN
        walls = breakdown.get("structure_walls", {})
        if walls and walls.get("total_cost", 0) > 0:
            story.append(Paragraph("2. Structural Walls Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Interior Wall Area (m²) × Interior Unit Price × Prefabrication Modifier + "
                "Exterior Wall Area (m²) × Exterior Unit Price × Prefabrication Modifier",
                styles["Body"]
            ))
            
            # Încarcă area_data și cubicasa_result pentru a afișa măsurătorile detaliate
            try:
                area_json_path = plan.stage_work_dir.parent.parent / "area" / plan.plan_id / "areas_calculated.json"
                cubicasa_json_path = plan.stage_work_dir / "cubicasa_result.json"
                
                area_data = None
                cubicasa_data = None
                
                if area_json_path.exists():
                    with open(area_json_path, "r", encoding="utf-8") as f:
                        area_data = json.load(f)
                
                if cubicasa_json_path.exists():
                    with open(cubicasa_json_path, "r", encoding="utf-8") as f:
                        cubicasa_data = json.load(f)
                    
                    # Adaugă secțiunea detaliată de calcul
                    story.append(Spacer(1, 2*mm))
                    story.append(Paragraph("<b>Detailed Calculation Process:</b>", styles["H3"]))
                    
                    # Scala
                    if cubicasa_data and "measurements" in cubicasa_data:
                        measurements = cubicasa_data["measurements"]
                        scale_m_px = measurements.get("metrics", {}).get("scale_m_per_px", 0.0)
                        pixels_data = measurements.get("pixels", {})
                        metrics_data = measurements.get("metrics", {})
                        
                        story.append(Paragraph(f"<b>Step 1: Scale Detection</b>", styles["Body"]))
                        story.append(Paragraph(
                            f"Scale determined from room labels: <b>{scale_m_px:.9f} m/pixel</b>",
                            styles["Body"]
                        ))
                        story.append(Spacer(1, 2*mm))
                        
                        # Pixeli din fiecare tip de imagine
                        story.append(Paragraph("<b>Step 2: Pixel Measurements from Images</b>", styles["Body"]))
                        story.append(Paragraph(
                            "The following pixel counts are extracted from different image types:",
                            styles["Body"]
                        ))
                        
                        px_ext = pixels_data.get("walls_len_ext", 0)
                        px_int = pixels_data.get("walls_len_int", 0)
                        px_skeleton_complete = pixels_data.get("walls_skeleton_complete", 0)
                        px_skeleton_ext = pixels_data.get("walls_skeleton_ext", 0)
                        px_skeleton_int = pixels_data.get("walls_skeleton_int", 0)
                        
                        story.append(Paragraph(
                            f"• <b>Exterior walls (blue outline):</b> {px_ext:,} pixels",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• <b>Interior walls (green outline):</b> {px_int:,} pixels",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• <b>Skeleton complete (all walls):</b> {px_skeleton_complete:,} pixels",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• <b>Skeleton exterior (from skeleton):</b> {px_skeleton_ext:,} pixels",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• <b>Skeleton interior (skeleton - exterior):</b> {px_skeleton_int:,} pixels",
                            styles["Body"]
                        ))
                        story.append(Spacer(1, 2*mm))
                        
                        # Conversia pixeli → metri
                        story.append(Paragraph("<b>Step 3: Conversion from Pixels to Meters</b>", styles["Body"]))
                        story.append(Paragraph(
                            "Each pixel count is multiplied by the scale to get length in meters:",
                            styles["Body"]
                        ))
                        
                        walls_ext_m = metrics_data.get("walls_ext_m", 0.0)
                        walls_int_m = metrics_data.get("walls_int_m", 0.0)
                        walls_skeleton_complete_m = metrics_data.get("walls_skeleton_complete_m", 0.0)
                        walls_skeleton_ext_m = metrics_data.get("walls_skeleton_ext_m", 0.0)
                        walls_skeleton_int_m = metrics_data.get("walls_skeleton_int_m", 0.0)
                        
                        story.append(Paragraph(
                            f"• Exterior (outline): {px_ext:,} px × {scale_m_px:.9f} m/px = <b>{walls_ext_m:.2f} m</b> (for finishes)",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• Interior (outline): {px_int:,} px × {scale_m_px:.9f} m/px = <b>{walls_int_m:.2f} m</b> (for finishes)",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• Skeleton complete: {px_skeleton_complete:,} px × {scale_m_px:.9f} m/px = <b>{walls_skeleton_complete_m:.2f} m</b>",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• Skeleton exterior: {px_skeleton_ext:,} px × {scale_m_px:.9f} m/px = <b>{walls_skeleton_ext_m:.2f} m</b>",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"• Skeleton interior: {px_skeleton_int:,} px × {scale_m_px:.9f} m/px = <b>{walls_skeleton_int_m:.2f} m</b> (for structure)",
                            styles["Body"]
                        ))
                        story.append(Spacer(1, 2*mm))
                    
                    # Calculele ariilor
                    if area_data:
                        story.append(Paragraph("<b>Step 4: Area Calculations</b>", styles["Body"]))
                        story.append(Paragraph(
                            "Wall areas are calculated by multiplying length by wall height, then subtracting openings:",
                            styles["Body"]
                        ))
                        
                        walls_data = area_data.get("walls", {})
                        interior_data = walls_data.get("interior", {})
                        exterior_data = walls_data.get("exterior", {})
                        
                        # Exterior
                        ext_length = exterior_data.get("length_m", 0.0)
                        ext_gross = exterior_data.get("gross_area_m2", 0.0)
                        ext_openings = exterior_data.get("openings_area_m2", 0.0)
                        ext_net = exterior_data.get("net_area_m2", 0.0)
                        wall_height = area_data.get("wall_height_m", 2.7)
                        
                        story.append(Paragraph(
                            f"<b>Exterior Walls:</b>",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"  • Length: {ext_length:.2f} m × Height: {wall_height:.2f} m = Gross Area: {ext_gross:.2f} m²",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"  • Gross Area: {ext_gross:.2f} m² - Openings: {ext_openings:.2f} m² = <b>Net Area: {ext_net:.2f} m²</b>",
                            styles["Body"]
                        ))
                        story.append(Spacer(1, 1*mm))
                        
                        # Interior finisaje
                        int_length_finish = interior_data.get("length_m", 0.0)
                        int_gross_finish = interior_data.get("gross_area_m2", 0.0)
                        int_openings = interior_data.get("openings_area_m2", 0.0)
                        int_net_finish = interior_data.get("net_area_m2", 0.0)
                        
                        story.append(Paragraph(
                            f"<b>Interior Walls (Finishes - from green outline):</b>",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"  • Length: {int_length_finish:.2f} m × Height: {wall_height:.2f} m = Gross Area: {int_gross_finish:.2f} m²",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"  • Gross Area: {int_gross_finish:.2f} m² - Openings: {int_openings:.2f} m² = <b>Net Area: {int_net_finish:.2f} m²</b>",
                            styles["Body"]
                        ))
                        story.append(Spacer(1, 1*mm))
                        
                        # Interior structură
                        int_length_structure = interior_data.get("length_m_structure", 0.0)
                        int_gross_structure = interior_data.get("gross_area_m2_structure", 0.0)
                        int_net_structure = interior_data.get("net_area_m2_structure", 0.0)
                        
                        story.append(Paragraph(
                            f"<b>Interior Walls (Structure - from skeleton):</b>",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"  • Length: {int_length_structure:.2f} m × Height: {wall_height:.2f} m = Gross Area: {int_gross_structure:.2f} m²",
                            styles["Body"]
                        ))
                        story.append(Paragraph(
                            f"  • Gross Area: {int_gross_structure:.2f} m² - Openings: {int_openings:.2f} m² = <b>Net Area: {int_net_structure:.2f} m²</b>",
                            styles["Body"]
                        ))
                        story.append(Spacer(1, 2*mm))
                    
                    # Adaugă tabelul cu măsurătorile pereților
                    if area_data:
                        from .tables import create_wall_measurements_table
                        measurements_table = create_wall_measurements_table(area_data, enforcer)
                        story.append(Paragraph("<b>Summary Table - Wall Measurements:</b>", styles["Body"]))
                        story.append(measurements_table)
                        story.append(Spacer(1, 2*mm))
                        
            except Exception as e:
                print(f"⚠️ [PDF] Could not load detailed measurements for {plan.plan_id}: {e}")
                import traceback
                traceback.print_exc()
            
            items = walls.get("detailed_items", [])
            for item in items:
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                name = item.get("name", "")
                material = item.get("material", "—")
                construction_mode = item.get("construction_mode", "—")
                
                story.append(Paragraph(
                    f"<b>Material Selected:</b> {material} | <b>Construction Mode:</b> {construction_mode}",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 3. OPENINGS (WINDOWS & DOORS) CALCULATION FOR THIS PLAN
        openings = breakdown.get("openings", {})
        if openings and openings.get("total_cost", 0) > 0:
            story.append(Paragraph("3. Openings (Windows & Doors) Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Opening Area (m²) × Unit Price per m² × Quality Multiplier (for windows)",
                styles["Body"]
            ))
            
            # Show user selections
            if bodentiefe_fenster and bodentiefe_fenster != "—":
                window_height_map = {
                    "Nein": "1.0m",
                    "Ja – einzelne": "1.5m",
                    "Ja – mehrere / große Glasflächen": "2.0m"
                }
                height = window_height_map.get(bodentiefe_fenster, "1.25m")
                story.append(Paragraph(
                    f"<b>Window Height Setting:</b> {bodentiefe_fenster} → All windows use height: {height}",
                    styles["Body"]
                ))
            
            if turhohe and turhohe != "—":
                door_height = "2.2m" if "Erhöht" in turhohe else "2.0m"
                story.append(Paragraph(
                    f"<b>Door Height Setting:</b> {turhohe} → All doors use height: {door_height}",
                    styles["Body"]
                ))
            
            if window_quality and window_quality != "—":
                quality_mult_map = {
                    "3-fach verglast": "1.25x",
                    "3-fach verglast, Passiv": "1.6x"
                }
                mult = quality_mult_map.get(window_quality, "1.0x")
                story.append(Paragraph(
                    f"<b>Window Quality:</b> {window_quality} → Multiplier: {mult}",
                    styles["Body"]
                ))
            
            story.append(Paragraph(
                f"<b>Window/Door Material:</b> Lemn-Aluminiu (standard, fixed)",
                styles["Body"]
            ))
            
            items = openings.get("detailed_items", [])
            story.append(Spacer(1, 2*mm))
            story.append(Paragraph("<b>All Openings for This Plan:</b>", styles["Body"]))
            for item in items:
                name = item.get("name", "")
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("total_cost", 0)
                dimensions = item.get("dimensions_m", "")
                material = item.get("material", "—")
                
                story.append(Paragraph(
                    f"{name} ({dimensions}, {material}): {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 4. FINISHES CALCULATION FOR THIS PLAN
        finishes = breakdown.get("finishes", {})
        if finishes and finishes.get("total_cost", 0) > 0:
            story.append(Paragraph("4. Finishes Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Wall Area (m²) × Finish Unit Price per m²",
                styles["Body"]
            ))
            
            items = finishes.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                material = item.get("material", "—")
                
                story.append(Paragraph(
                    f"<b>Finish Selected:</b> {material}",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 5. FLOORS & CEILINGS CALCULATION FOR THIS PLAN
        floors = breakdown.get("floors_ceilings", {})
        if floors and floors.get("total_cost", 0) > 0:
            story.append(Paragraph("5. Floors & Ceilings Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Floor Area (m²) × Floor Coefficient + Ceiling Area (m²) × Ceiling Coefficient",
                styles["Body"]
            ))
            
            area_coeffs = pricing_coeffs.get("area", {})
            floor_coeff = area_coeffs.get("floor_coefficient_per_m2", 0)
            ceiling_coeff = area_coeffs.get("ceiling_coefficient_per_m2", 0)
            
            story.append(Paragraph(f"<b>Floor Coefficient Used:</b> {floor_coeff:.2f} EUR/m²", styles["Body"]))
            story.append(Paragraph(f"<b>Ceiling Coefficient Used:</b> {ceiling_coeff:.2f} EUR/m²", styles["Body"]))
            
            items = floors.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 6. UTILITIES CALCULATION FOR THIS PLAN (only for ground floor)
        utilities = breakdown.get("utilities", {})
        if utilities and utilities.get("total_cost", 0) > 0:
            story.append(Paragraph("6. Utilities & Installations Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Total Floor Area (m²) × Base Coefficient × Energy Modifier × Type Modifier",
                styles["Body"]
            ))
            
            # Show user selections
            if nivel_energetic and nivel_energetic != "—":
                story.append(Paragraph(
                    f"<b>Energy Level Selected:</b> {nivel_energetic}",
                    styles["Body"]
                ))
            if tip_incalzire and tip_incalzire != "—":
                story.append(Paragraph(
                    f"<b>Heating Type Selected:</b> {tip_incalzire}",
                    styles["Body"]
                ))
            story.append(Paragraph(
                f"<b>Ventilation Selected:</b> {'Yes' if ventilatie else 'No'}",
                styles["Body"]
            ))
            
            items = utilities.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                area = item.get("area_m2", 0)
                base_price = item.get("base_price_per_m2", 0)
                final_price = item.get("final_price_per_m2", 0)
                cost = item.get("total_cost", 0)
                energy_mod = item.get("energy_modifier", 1.0)
                type_mod = item.get("type_modifier", 1.0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}",
                    styles["Body"]
                ))
                story.append(Paragraph(
                    f"  Base: {base_price:.2f} EUR/m² × Energy Modifier: {energy_mod:.2f}x × Type Modifier: {type_mod:.2f}x = {final_price:.2f} EUR/m²",
                    styles["Small"]
                ))
                story.append(Paragraph(
                    f"  {area:.2f} m² × {final_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 7. FIREPLACE & CHIMNEY CALCULATION FOR THIS PLAN (only for ground floor)
        fireplace = breakdown.get("fireplace", {})
        if fireplace and fireplace.get("total_cost", 0) > 0:
            story.append(Paragraph("7. Fireplace & Chimney Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Fireplace Cost = 4,500 EUR (fixed) + Chimney Cost = 1,500 EUR × Number of Floors",
                styles["Body"]
            ))
            
            story.append(Paragraph(
                f"<b>Fireplace Selected:</b> {'Yes' if semineu else 'No'}",
                styles["Body"]
            ))
            
            items = fireplace.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                unit_price = item.get("unit_price", 0)
                quantity = item.get("quantity", 1)
                cost = item.get("total_cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {quantity} × {unit_price:.2f} EUR = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 8. STAIRS CALCULATION FOR THIS PLAN
        stairs = breakdown.get("stairs", {})
        if stairs and stairs.get("total_cost", 0) > 0:
            story.append(Paragraph("8. Stairs Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Number of Stair Units × Price per Unit + Railing Cost",
                styles["Body"]
            ))
            
            stairs_coeffs = pricing_coeffs.get("stairs", {})
            stair_unit_price = stairs_coeffs.get("price_per_stair_unit", 0)
            railing_price = stairs_coeffs.get("railing_price_per_stair", 0)
            
            story.append(Paragraph(f"<b>Stair Unit Price Used:</b> {stair_unit_price:.2f} EUR/unit", styles["Body"]))
            story.append(Paragraph(f"<b>Railing Price Used:</b> {railing_price:.2f} EUR/stair", styles["Body"]))
            
            items = stairs.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                quantity = item.get("quantity", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {quantity} × {unit_price:.2f} EUR = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
    
        # 9. ROOF CALCULATION FOR THIS PLAN
        roof = breakdown.get("roof", {})
        if roof and roof.get("total_cost", 0) > 0:
            story.append(Paragraph("9. Roof Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Cost = Roof Area (m²) × Material Unit Price per m²",
                styles["Body"]
            ))
            
            if tip_acoperis and tip_acoperis != "—":
                story.append(Paragraph(
                    f"<b>Roof Type Selected:</b> {tip_acoperis}",
                    styles["Body"]
                ))
            
            material_acoperis = materiale_finisaj.get("materialAcoperis", "—")
            if material_acoperis and material_acoperis != "—":
                story.append(Paragraph(
                    f"<b>Roof Material Selected:</b> {material_acoperis}",
                    styles["Body"]
                ))
            
            items = roof.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                area = item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {area:.2f} m² × {unit_price:.2f} EUR/m² = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
        
        # 9.5. BASEMENT CALCULATION FOR THIS PLAN (if exists)
        basement = breakdown.get("basement", {})
        if basement and basement.get("total_cost", 0) > 0:
            story.append(Paragraph("9.5. Basement Calculation", styles["H2"]))
            story.append(Paragraph(
                "<b>Formula:</b> Basement Cost = Interior Walls + Floors + Finishes + Utilities (if livable)",
                styles["Body"]
            ))
            story.append(Paragraph(
                "Basement is calculated using the same areas as ground floor, but only interior walls (no exterior walls).",
                styles["Small"]
            ))
            
            items = basement.get("detailed_items", [])
            for item in items:
                name = item.get("name", "")
                quantity = item.get("quantity", 0) or item.get("area_m2", 0)
                unit_price = item.get("unit_price", 0)
                cost = item.get("cost", 0) or item.get("total_cost", 0)
                
                story.append(Paragraph(
                    f"<b>Calculation:</b> {name}: {quantity:.2f} × {unit_price:.2f} EUR = <b>{cost:.2f} EUR</b>",
                    styles["Body"]
                ))
            
            story.append(Spacer(1, 4*mm))
        
        # Plan total
        plan_total = pricing.get("total_cost_eur", 0.0)
        story.append(Paragraph(
            f"<b>Total Cost for This Plan:</b> {plan_total:,.2f} EUR",
            styles["H2"]
        ))
        story.append(Spacer(1, 6*mm))
    
    story.append(PageBreak())
    
    # 10. TOTAL COST CALCULATION
    story.append(Paragraph("10. Total Cost Calculation", styles["H2"]))
    story.append(Paragraph(
        "<b>Formula:</b> Gross Total = (Construction Costs + Logistics Margin + Oversight Margin) × (1 + VAT Rate)",
        styles["Body"]
    ))
    story.append(Paragraph(
        "• Construction Costs: Sum of all component costs (foundation, walls, openings, finishes, etc.)",
        styles["Small"]
    ))
    story.append(Paragraph(
        "• Logistics Margin: 10% of construction costs",
        styles["Small"]
    ))
    story.append(Paragraph(
        "• Oversight Margin: 10% of construction costs",
        styles["Small"]
    ))
    story.append(Paragraph(
        "• VAT Rate: 19% (applied to net total)",
        styles["Small"]
    ))
    
    # Calculate totals from actual data
    total_construction = sum(e["pricing"].get("total_cost_eur", 0.0) for e in plans_data)
    logistics_margin = total_construction * 0.10
    oversight_margin = total_construction * 0.10
    net_total = total_construction + logistics_margin + oversight_margin
    vat = net_total * 0.19
    gross_total = net_total + vat
    
    story.append(Spacer(1, 4*mm))
    story.append(Paragraph("<b>Actual Calculation for This Project:</b>", styles["Body"]))
    story.append(Paragraph(
        f"Construction Costs: {total_construction:,.2f} EUR",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"Logistics Margin (10%): {logistics_margin:,.2f} EUR",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"Oversight Margin (10%): {oversight_margin:,.2f} EUR",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"<b>Net Total (excl. VAT):</b> {net_total:,.2f} EUR",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"VAT (19%): {vat:,.2f} EUR",
        styles["Body"]
    ))
    story.append(Paragraph(
        f"<b>Gross Total (incl. VAT):</b> {gross_total:,.2f} EUR",
        styles["H2"]
    ))
    
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph("END OF CALCULATION METHOD DOCUMENTATION", styles["Small"]))
    
    doc.build(story)
    
    print(f"✅ [PDF CALC METHOD] Generat: {output_path}")
    return output_path