from typing import List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.laminate_failure_comparison import evaluate_laminate_failure


app = FastAPI(
    title="CFRP Failure Prediction API",
    description=(
        "API for Classical Lamination Theory and composite laminate "
        "failure prediction."
    ),
    version="1.0.0",
)


# ============================================================
# Input schemas
# ============================================================

class MaterialProperties(BaseModel):
    E1: float = Field(..., gt=0, description="Longitudinal modulus (Pa)")
    E2: float = Field(..., gt=0, description="Transverse modulus (Pa)")
    G12: float = Field(..., gt=0, description="In-plane shear modulus (Pa)")
    nu12: float = Field(..., description="Major Poisson's ratio")


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


@app.post("/analyze")
def analyze_laminate(request: AnalysisRequest):
    """
    Run CLT laminate analysis and evaluate all validated
    composite failure criteria.
    """

    material = request.material
    strengths = request.strengths
    loads = request.loads

    # --------------------------------------------------------
    # Convert API input into physics-engine inputs
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Run validated laminate physics engine
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Find governing result for each criterion
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Serialize ply results
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Return JSON response
    # --------------------------------------------------------

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
