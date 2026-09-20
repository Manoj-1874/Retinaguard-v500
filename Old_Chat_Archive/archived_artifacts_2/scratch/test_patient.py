import sys
sys.path.append(r'e:\V500')
from patient_history_module import PatientHistoryModule

patient_data_raw = {
    "symptoms": {
        "nightBlindness": False,
        "tunnelVision": False,
        "colorVisionLoss": False,
        "glareSensitivity": False
    },
    "age": 45,
    "durationYears": 0,
    "familyHistory": False
}

module = PatientHistoryModule()
patient_data = module.collect_patient_data(patient_data_raw)
print(patient_data['threshold_adjustments'])
