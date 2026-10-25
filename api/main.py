from typing import List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tensorflow import keras
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.laminate_failure_comparison import evaluate_laminate_failure
from src.laminate_failure_envelope import find_first_ply_failure
from src.laminate_loading_study import run_loading_study
from src.materials import FIBERS, MATRICES, get_fiber, get_matrix
from src.micromechanics import calculate_lamina_properties
from src.strengths import (
    LAMINA_STRENGTHS,
    find_lamina_strengths,
)


app = FastAPI(
    title="CFRP Failure Prediction API",
    description=(
        "API for Classical Lamination Theory and composite laminate "
        "failure prediction."
    ),
    version="1.0.0",
)



app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "https://cfrp-ml-failure-prediction-5wy06fxak-akhbv.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




# ============================================================
# ML surrogate models - V5, seven targets
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ML_SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_multitarget_v5"
)

ML_RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "multitarget_dnn_v5"
)

ML_TARGETS = {
    "max_stress_fi": "Maximum Stress",
    "max_strain_fi": "Maximum Strain",
    "tsai_hill_fi": "Tsai-Hill",
    "tsai_wu_fi": "Tsai-Wu",
    "hoffman_fi": "Hoffman",
    "hashin_fi": "Hashin",
    "puck_fi": "Puck",
}

ML_FEATURES = ["Nx", "Ny", "Nxy", "Mx", "My", "Mxy"]

ML_FAILURE_THRESHOLD = 1.0

ml_scaler = joblib.load(
    ML_SPLIT_DIR / "feature_scaler.joblib"
)

ml_models = {
    target: keras.models.load_model(
        ML_RESULTS_DIR / target / f"{target}_model.keras"
    )
    for target in ML_TARGETS
}


# ============================================================
# Input schemas
# ============================================================

class MaterialProperties(BaseModel):
    E1: float = Field(..., gt=0, description="Longitudinal modulus (Pa)")
    E2: float = Field(..., gt=0, description="Transverse modulus (Pa)")
    G12: float = Field(..., gt=0, description="In-plane shear modulus (Pa)")
    nu12: float = Field(..., description="Major Poisson's ratio")

class MaterialSelection(BaseModel):
    fiber: str
    matrix: str
    Vf: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Fiber volume fraction"
    )


class Strengths(BaseModel):
    Xt: float = Field(..., gt=0, description="Longitudinal tensile strength (Pa)")
    Xc: float = Field(..., gt=0, description="Longitudinal compressive strength (Pa)")
    Yt: float = Field(..., gt=0, description="Transverse tensile strength (Pa)")
    Yc: float = Field(..., gt=0, description="Transverse compressive strength (Pa)")
    S: float = Field(..., gt=0, description="In-plane shear strength (Pa)")


class Loads(BaseModel):
    Nx: float = Field(0.0, description="In-plane force resultant Nx (N/m)")
    Ny: float = Field(0.0, description="In-plane force resultant Ny (N/m)")
    Nxy: float = Field(0.0, description="In-plane shear resultant Nxy (N/m)")
    Mx: float = Field(0.0, description="Moment resultant Mx (N)")
    My: float = Field(0.0, description="Moment resultant My (N)")
    Mxy: float = Field(0.0, description="Twisting moment resultant Mxy (N)")

class MLPredictionRequest(BaseModel):
    Nx: float = 0.0
    Ny: float = 0.0
    Nxy: float = 0.0
    Mx: float = 0.0
    My: float = 0.0
    Mxy: float = 0.0

class AnalysisRequest(BaseModel):
    material: MaterialProperties
    strengths: Strengths
    layup: List[float] = Field(..., min_length=1)
    ply_thickness: float = Field(
        ...,
        gt=0,
        description="Thickness of each ply (m)",
    )
    loads: Loads


# ============================================================
# Utility functions
# ============================================================

def numpy_to_list(value):
    """Convert NumPy arrays/scalars into JSON-safe Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    return value


def serialize_criterion_results(criteria):
    """Convert all criterion outputs into JSON-safe values."""
    serialized = {}

    for criterion_name, result in criteria.items():
        serialized[criterion_name] = {}

        for key, value in result.items():
            serialized[criterion_name][key] = numpy_to_list(value)

    return serialized


# ============================================================
# Routes
# ============================================================


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "message": "CFRP Failure Prediction API is running.",
    }



@app.post("/ml-predict")
def ml_predict(request: MLPredictionRequest):
    """
    Predict seven failure indices using the corrected v2 DNN
    surrogate models.

    Decision rule:
        FI < 1  -> SAFE
        FI >= 1 -> FAILURE

    Predictions are surrogate estimates, not experimental validation.
    """

    features = np.array(
        [[
            request.Nx,
            request.Ny,
            request.Nxy,
            request.Mx,
            request.My,
            request.Mxy,
        ]],
        dtype=np.float32,
    )

    scaled_features = ml_scaler.transform(features)

    predictions = {}

    for target, display_name in ML_TARGETS.items():
        predicted_fi = float(
            ml_models[target](scaled_features, training=False)
            .numpy().reshape(-1)[0]
        )


        # Failure indices are physically non-negative.
        predicted_fi = max(0.0, predicted_fi)
        failed = predicted_fi >= ML_FAILURE_THRESHOLD

        predictions[target] = {
            "name": display_name,
            "failure_index": predicted_fi,
            "failed": failed,
            "decision": "FAILURE" if failed else "SAFE",
        }

    # Preserve the original response field expected by the current UI.
    hashin_prediction = predictions["hashin_fi"]

    return {
        "status": "success",
        "model": {
            "type": "DNN regression surrogate",
            "version": "v5",
            "architecture": "[64, 64, 32]",
            "targets": list(ML_TARGETS.keys()),
            "failure_threshold": ML_FAILURE_THRESHOLD,
            "experimental_validation": "pending",
        },
        "prediction": {
            "hashin_fi": hashin_prediction["failure_index"],
            "failed": hashin_prediction["failed"],
            "decision": hashin_prediction["decision"],
        },
        "predictions": predictions,
    }


@app.get("/materials")
def get_available_materials():
    carbon_fibers = ["T300", "AS4", "IM7"]

    compatible_systems = [
        {
            "fiber": data["fiber"],
            "matrix": data["matrix"],
            "system_id": system_id,
            "system_name": data["system_name"],
        }
        for system_id, data in LAMINA_STRENGTHS.items()
        if data["fiber"] in carbon_fibers
    ]

    available_fibers = [
        fiber for fiber in carbon_fibers
        if fiber in FIBERS
    ]

    available_matrices = list(dict.fromkeys(
        system["matrix"] for system in compatible_systems
    ))

    return {
        "fibers": available_fibers,
        "matrices": available_matrices,
        "compatible_systems": compatible_systems,
    }



@app.post("/material-properties")
def calculate_material_properties(selection: MaterialSelection):
    try:
        fiber = get_fiber(selection.fiber)
        matrix = get_matrix(selection.matrix)

        E1, E2, G12, nu12 = calculate_lamina_properties(
            Ef=fiber["E"],
            Em=matrix["E"],
            Gf=fiber["G"],
            Gm=matrix["G"],
            nu_f=fiber["nu"],
            nu_m=matrix["nu"],
            Vf=selection.Vf,
        )

        return {
            "status": "success",
            "fiber": selection.fiber,
            "matrix": selection.matrix,
            "fiber_volume_fraction": selection.Vf,
            "properties": {
                "E1": E1,
                "E2": E2,
                "G12": G12,
                "nu12": nu12,
            },
        }

    except KeyError as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


@app.get("/strengths")
def get_available_strengths():
    return {
        "systems": list(LAMINA_STRENGTHS.keys())
    }


@app.get("/material-strengths")
def get_material_strengths(fiber: str, matrix: str):
    strengths = find_lamina_strengths(fiber, matrix)

    if strengths is None:
        return {
            "status": "unavailable",
            "fiber": fiber,
            "matrix": matrix,
            "message": (
                "No verified lamina strength dataset is available "
                "for this material combination."
            ),
        }

    return {
        "status": "success",
        "fiber": fiber,
        "matrix": matrix,
        "system_id": strengths["system_id"],
        "system_name": strengths["system_name"],
        "strengths": {
            "Xt": strengths["Xt"],
            "Xc": strengths["Xc"],
            "Yt": strengths["Yt"],
            "Yc": strengths["Yc"],
            "S": strengths["S"],
        },
        "source": strengths["source"],
        "source_type": strengths["source_type"],
        "notes": strengths["notes"],
    }

@app.post("/analyze")
def analyze_laminate(request: AnalysisRequest):
    """
    Run CLT laminate analysis and evaluate all validated
    composite failure criteria.
    """

    material = request.material
    strengths = request.strengths
    loads = request.loads

    N = np.array([
        loads.Nx,
        loads.Ny,
        loads.Nxy,
    ])

    M = np.array([
        loads.Mx,
        loads.My,
        loads.Mxy,
    ])

    strengths_dict = {
        "Xt": strengths.Xt,
        "Xc": strengths.Xc,
        "Yt": strengths.Yt,
        "Yc": strengths.Yc,
        "S": strengths.S,
    }

    results = evaluate_laminate_failure(
        E1=material.E1,
        E2=material.E2,
        G12=material.G12,
        nu12=material.nu12,
        ply_angles=request.layup,
        ply_thickness=request.ply_thickness,
        N=N,
        M=M,
        strengths=strengths_dict,
    )

    criterion_names = [
        "Maximum Stress",
        "Maximum Strain",
        "Tsai-Hill",
        "Tsai-Wu",
        "Hoffman",
        "Hashin",
        "Puck",
    ]

    governing_results = {}

    for criterion in criterion_names:
        governing = max(
            results["ply_results"],
            key=lambda result: result["criteria"][criterion]["failure_index"],
        )

        governing_results[criterion] = {
            "failure_index": float(
                governing["criteria"][criterion]["failure_index"]
            ),
            "failed": bool(
                governing["criteria"][criterion]["failed"]
            ),
            "ply": int(governing["ply"]),
            "angle": float(governing["angle"]),
            "surface": governing["surface"],
        }

        if "failure_mode" in governing["criteria"][criterion]:
            governing_results[criterion]["failure_mode"] = (
                governing["criteria"][criterion]["failure_mode"]
            )

    ply_results = []

    for result in results["ply_results"]:
        ply_results.append({
            "ply": int(result["ply"]),
            "angle": float(result["angle"]),
            "surface": result["surface"],
            "strain_global": numpy_to_list(result["strain_global"]),
            "strain_local": numpy_to_list(result["strain_local"]),
            "stress_local": numpy_to_list(result["stress_local"]),
            "criteria": serialize_criterion_results(
                result["criteria"]
            ),
        })

    return {
        "status": "success",

        "laminate": {
            "number_of_plies": len(request.layup),
            "layup": request.layup,
            "ply_thickness_m": request.ply_thickness,
            "total_thickness_m": (
                len(request.layup) * request.ply_thickness
            ),
        },

        "loads": {
            "N": N.tolist(),
            "M": M.tolist(),
        },

        "ABD": {
            "A": numpy_to_list(results["A"]),
            "B": numpy_to_list(results["B"]),
            "D": numpy_to_list(results["D"]),
        },

        "response": {
            "mid_plane_strain": numpy_to_list(
                results["mid_plane_strain"]
            ),
            "curvature": numpy_to_list(
                results["curvature"]
            ),
        },

        "governing_results": governing_results,

        "ply_results": ply_results,
    }


@app.post("/first-ply-failure")
def first_ply_failure(request: AnalysisRequest):
    """
    Determine the load factor at which first-ply failure occurs
    for each validated failure criterion.
    """

    material = request.material
    strengths = request.strengths
    loads = request.loads

    N_base = np.array([
        loads.Nx,
        loads.Ny,
        loads.Nxy,
    ])

    M_base = np.array([
        loads.Mx,
        loads.My,
        loads.Mxy,
    ])

    strengths_dict = {
        "Xt": strengths.Xt,
        "Xc": strengths.Xc,
        "Yt": strengths.Yt,
        "Yc": strengths.Yc,
        "S": strengths.S,
    }

    results = find_first_ply_failure(
        E1=material.E1,
        E2=material.E2,
        G12=material.G12,
        nu12=material.nu12,
        ply_angles=request.layup,
        ply_thickness=request.ply_thickness,
        N_base=N_base,
        M_base=M_base,
        strengths=strengths_dict,
    )

    serialized_results = {}

    for criterion, result in results.items():
        serialized_results[criterion] = {
            "critical_load_factor": float(
                result["critical_load_factor"]
            ),
            "failure_index": float(
                result["failure_index"]
            ),
            "ply": int(result["ply"]),
            "angle": float(result["angle"]),
            "surface": result["surface"],
            "failed": bool(result["failed"]),
        }

        if "failure_mode" in result:
            serialized_results[criterion]["failure_mode"] = (
                result["failure_mode"]
            )

    return {
        "status": "success",
        "message": (
            "First-ply-failure load factors calculated "
            "for all validated criteria."
        ),
        "load_definition": {
            "N_base": N_base.tolist(),
            "M_base": M_base.tolist(),
            "meaning": (
                "Critical load factor lambda scales the supplied "
                "base laminate loads."
            ),
        },
        "results": serialized_results,
    }

@app.post("/loading-study")
def loading_study(request: AnalysisRequest):
    """
    Run the validated multi-loading first-ply-failure study.

    Uses the default six loading cases defined in
    src.laminate_loading_study.py.
    """

    strengths_dict = {
        "Xt": request.strengths.Xt,
        "Xc": request.strengths.Xc,
        "Yt": request.strengths.Yt,
        "Yc": request.strengths.Yc,
        "S": request.strengths.S,
    }

    results = run_loading_study(
        ply_angles=request.layup,
        ply_thickness=request.ply_thickness,
        strengths=strengths_dict,
        E1=request.material.E1,
        E2=request.material.E2,
        G12=request.material.G12,
        nu12=request.material.nu12,
    )

    records = []

    for _, row in results.iterrows():
        records.append({
            "loading_case": row["loading_case"],
            "criterion": row["criterion"],
            "base_loads": {
                "Nx_N_per_m": float(row["base_Nx_N_per_m"]),
                "Ny_N_per_m": float(row["base_Ny_N_per_m"]),
                "Nxy_N_per_m": float(row["base_Nxy_N_per_m"]),
            },
            "critical_load_factor": float(row["critical_load_factor"]),
            "critical_loads": {
                "Nx_N_per_m": float(row["critical_Nx_N_per_m"]),
                "Ny_N_per_m": float(row["critical_Ny_N_per_m"]),
                "Nxy_N_per_m": float(row["critical_Nxy_N_per_m"]),
            },
            "failure_index": float(row["failure_index"]),
            "ply": int(row["ply"]),
            "angle_deg": float(row["angle_deg"]),
            "surface": row["surface"],
            "failure_mode": (

             None
             if pd.isna(row["failure_mode"])
             else row["failure_mode"]
            ),
        })

    return {
        "status": "success",
        "message": "Multi-loading first-ply-failure study completed.",
        "number_of_loading_cases": 6,
        "number_of_criteria": 7,
        "total_results": len(records),
        "results": records,
    }
