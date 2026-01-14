#---------------------------------------------
#CADRAN 1 - complet

EVALUATE HOUSE PLAN
python evaluate_plan/evaluate_house_plan.py

ADAUGARE EXEMPLE ROBOFLOW
python export_objects/import_detections.py

RULARE DETECTIE USI/GEAMURI
python export_objects/run_crops.py

RULARE NUMARARE USI/GEAMURI
python count_objects/detect_all_hybrid.py

CALCULARE METERS/PIXEL
python meters_pixel/analyze_scale.py

#FINAL CADRAN 1
#--------------------------------------------
#CADRAN 2 - complet

MEASURE DOORS AND WINDOWS IN METERS
python measure_objects/measure_openings.py

EXTRACT NUMBER OF EXTERIOR DOORS
python exterior_doors/room_extraction.py
python exterior_doors/detect_exterior_doors.py

#FINAL CADRAN 2
#--------------------------------------------
#CADRAN 3 - complet

MASURARE LUNGIME DESCHIDERI
python perimeter/openings_data.py

MASURARE PERETI INTERIORI SI EXTERIORI
python perimeter/measure_walls.py

#FINAL CADRAN 3
#--------------------------------------------
CADRAN 4 - complet

CALCULARE ARIE FINALA PERETI
python area/calculate_wall_areas.py

#FINAL CADRAN 4
#--------------------------------------------
#CADRAN 5 - complet

CALCULARE ARIA CASEI
python area/calculate_total_area_gemini.py

#FINAL CADRAN 5
#--------------------------------------------
CADRAN 6 - complet

CALCULARE PRET ACOPERIS
python roof/calculate_roof_price.py

#FINAL CADRAN 6