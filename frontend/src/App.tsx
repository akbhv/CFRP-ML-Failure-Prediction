import { useEffect, useState } from "react";
import { Activity, Layers3, Play, AlertCircle } from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

const UNITS = {
  metric: {
    thickness: "mm",
    modulus: "GPa",
    strength: "MPa",
    load: "N/m",
    moment: "N",
  },

  imperial: {
    thickness: "in",
    modulus: "Msi",
    strength: "ksi",
    load: "lbf/in",
    moment: "lbf",
  },
};

const CONVERSIONS = {
  metric: {
    modulusToSI: 1e9,       // GPa → Pa
    strengthToSI: 1e6,       // MPa → Pa
    thicknessToSI: 1e-3,     // mm → m
    loadToSI: 1,             // N/m → N/m
    momentToSI: 1,            // N → N

    modulusFromSI: 1e-9,      // Pa → GPa
    strengthFromSI: 1e-6,     // Pa → MPa
    thicknessFromSI: 1e3,     // m → mm
    loadFromSI: 1,            // N/m → N/m
    momentFromSI: 1,           // N → N
  },

  imperial: {
    // Modulus
    modulusToSI: 6894757293.168,
    modulusFromSI: 1 / 6894757293.168,

    // Strength
    strengthToSI: 6894757.293168,
    strengthFromSI: 1 / 6894757.293168,

    // Thickness
    thicknessToSI: 0.0254,
    thicknessFromSI: 1 / 0.0254,

    // Laminate load resultant
    // lbf/in → N/m
    loadToSI: 175.126835,
    loadFromSI: 1 / 175.126835,

    // Moment resultant
    // lbf → N
    momentToSI: 4.4482216152605,
    momentFromSI: 1 / 4.4482216152605,
  },
};

// ============================================================
// UNIT CONVERSIONS
// Frontend uses engineering units.
// Backend / physics engine uses SI units.
// ============================================================

const toSI = {
  modulus: (value: string | number, system: UnitSystem) =>
    Number(value) * CONVERSIONS[system].modulusToSI,

  strength: (value: string | number, system: UnitSystem) =>
    Number(value) * CONVERSIONS[system].strengthToSI,

  thickness: (value: string | number, system: UnitSystem) =>
    Number(value) * CONVERSIONS[system].thicknessToSI,

  load: (value: string | number, system: UnitSystem) =>
    Number(value) * CONVERSIONS[system].loadToSI,

  moment: (value: string | number, system: UnitSystem) =>
    Number(value) * CONVERSIONS[system].momentToSI,
};

const fromSI = {
  modulus: (value: number, system: UnitSystem) =>
    value * CONVERSIONS[system].modulusFromSI,

  strength: (value: number, system: UnitSystem) =>
    value * CONVERSIONS[system].strengthFromSI,

  thickness: (value: number, system: UnitSystem) =>
    value * CONVERSIONS[system].thicknessFromSI,

  load: (value: number, system: UnitSystem) =>
    value * CONVERSIONS[system].loadFromSI,

  moment: (value: number, system: UnitSystem) =>
    value * CONVERSIONS[system].momentFromSI,
};

const convertUnitValue = (
  value: string,
  type: "modulus" | "strength" | "thickness" | "load" | "moment",
  from: UnitSystem,
  to: UnitSystem
) => {
  if (from === to) {
    return value;
  }

  const siValue = toSI[type](value, from);
  const convertedValue = fromSI[type](siValue, to);

  return convertedValue.toString();
};

type GoverningResult = {
  failure_index: number;
  ply: number;
  angle: number;
  surface: string;
  failure_mode?: string | null;
};

type AnalysisResponse = {
  laminate: {
    number_of_plies: number;
    layup: number[];
    ply_thickness_m: number;
    total_thickness_m: number;
  };
  loads: {
    N: number[];
    M: number[];
  };
  ABD: {
    A: number[][];
    B: number[][];
    D: number[][];
  };
  response: {
    mid_plane_strain: number[];
    curvature: number[];
  };
  governing_results: Record<string, GoverningResult>;
};

type MaterialPropertiesResponse = {
  status: string;
  fiber: string;
  matrix: string;
  fiber_volume_fraction: number;
  properties: {
    E1: number;
    E2: number;
    G12: number;
    nu12: number;
  };
};

type MaterialsResponse = {
  fibers: string[];
  matrices: string[];
};

type UnitSystem = "metric" | "imperial";

function App() {
  const [status, setStatus] = useState("Ready");
  const [unitSystem, setUnitSystem] =
    useState<UnitSystem>("metric");
  const [loading, setLoading] = useState(false);
  const [materialLoading, setMaterialLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<AnalysisResponse | null>(null);

  // ============================================================
  // MATERIAL SELECTION
  // ============================================================

  const [fibers, setFibers] = useState<string[]>([]);
  const [matrices, setMatrices] = useState<string[]>([]);

  const [fiber, setFiber] = useState("T300");
  const [matrix, setMatrix] = useState("Standard Epoxy");
  const [Vf, setVf] = useState("0.60");

  // ============================================================
  // CALCULATED LAMINA PROPERTIES
  // ============================================================

  const [E1, setE1] = useState("139.6000");
  const [E2, setE2] = useState("9.7458");
  const [G12, setG12] = useState("3.2249");
  const [nu12, setNu12] = useState("0.2600");

  // ============================================================
  // LAMINA STRENGTHS
  // ============================================================

  const [Xt, setXt] = useState("1950");
  const [Xc, setXc] = useState("1480");
  const [Yt, setYt] = useState("48");
  const [Yc, setYc] = useState("200");
  const [S, setS] = useState("79");

  // ============================================================
  // LAMINATE
  // ============================================================

  const [plyThickness, setPlyThickness] = useState("0.125");

  const [layup, setLayup] = useState<number[]>([
    0,
    45,
    -45,
    90,
    90,
    -45,
    45,
    0,
  ]);

  // ============================================================
  // LOADS
  // ============================================================

  const [Nx, setNx] = useState("0");
  const [Ny, setNy] = useState("0");
  const [Nxy, setNxy] = useState("0");

  // ============================================================
  // LAYUP FUNCTIONS
  // ============================================================

  const updatePlyAngle = (index: number, value: string) => {
    const angle = Number(value);

    if (!Number.isFinite(angle)) {
      return;
    }

    setLayup((currentLayup) =>
      currentLayup.map((plyAngle, plyIndex) =>
        plyIndex === index ? angle : plyAngle
      )
    );
  };

  const addPly = () => {
    setLayup((currentLayup) => [...currentLayup, 0]);
  };

  const removePly = (index: number) => {
    setLayup((currentLayup) => {
      if (currentLayup.length <= 1) {
        return currentLayup;
      }

      return currentLayup.filter(
        (_, plyIndex) => plyIndex !== index
      );
    });
  };

  const resetLayup = () => {
    setLayup([
      0,
      45,
      -45,
      90,
      90,
      -45,
      45,
      0,
    ]);
  };

  const handleUnitSystemChange = (newSystem: UnitSystem) => {
    if (newSystem === unitSystem) {
      return;
    }

    // Strengths
    setXt(
      convertUnitValue(
        Xt,
        "strength",
        unitSystem,
        newSystem
      )
    );

    setXc(
      convertUnitValue(
        Xc,
        "strength",
        unitSystem,
        newSystem
      )
    );

    setYt(
      convertUnitValue(
        Yt,
        "strength",
        unitSystem,
        newSystem
      )
    );

    setYc(
      convertUnitValue(
        Yc,
        "strength",
        unitSystem,
        newSystem
      )
    );

    setS(
      convertUnitValue(
        S,
        "strength",
        unitSystem,
        newSystem
      )
    );

    // Ply thickness
    setPlyThickness(
      convertUnitValue(
        plyThickness,
        "thickness",
        unitSystem,
        newSystem
      )
    );

    // Applied loads
    setNx(
      convertUnitValue(
        Nx,
        "load",
        unitSystem,
        newSystem
      )
    );

    setNy(
      convertUnitValue(
        Ny,
        "load",
        unitSystem,
        newSystem
      )
    );

    setNxy(
      convertUnitValue(
        Nxy,
        "load",
        unitSystem,
        newSystem
      )
    );

    setUnitSystem(newSystem);
  };

  // ============================================================
  // LOAD MATERIAL DATABASE
  // ============================================================

  useEffect(() => {
    const loadMaterials = async () => {
      try {
        const response = await fetch(`${API_URL}/materials`);

        if (!response.ok) {
          throw new Error(
            `Failed to load materials: ${response.status}`
          );
        }

        const data: MaterialsResponse =
          await response.json();

        setFibers(data.fibers);
        setMatrices(data.matrices);

        if (
          data.fibers.length > 0 &&
          !data.fibers.includes(fiber)
        ) {
          setFiber(data.fibers[0]);
        }

        if (
          data.matrices.length > 0 &&
          !data.matrices.includes(matrix)
        ) {
          setMatrix(data.matrices[0]);
        }
      } catch (err) {
        console.error(err);

        setError(
          "Could not load the material database. Make sure the FastAPI server is running."
        );
      }
    };

    loadMaterials();
  }, []);

  // ============================================================
  // CALCULATE EFFECTIVE LAMINA PROPERTIES
  // ============================================================

  useEffect(() => {
    const calculateMaterialProperties = async () => {
      const vf = Number(Vf);

      if (
        !fiber ||
        !matrix ||
        !Number.isFinite(vf) ||
        vf <= 0 ||
        vf >= 1
      ) {
        return;
      }

      setMaterialLoading(true);

      try {
        const response = await fetch(
          `${API_URL}/material-properties`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              fiber,
              matrix,
              Vf: vf,
            }),
          }
        );

        if (!response.ok) {
          const message = await response.text();

          throw new Error(
            `Material API returned ${response.status}: ${message}`
          );
        }

        const data: MaterialPropertiesResponse =
          await response.json();

        if (data.status !== "success") {
          throw new Error(
            "Material-property calculation failed."
          );
        }

        setE1(
          fromSI.modulus(
            data.properties.E1,
            unitSystem
          ).toFixed(4)
        );

        setE2(
          fromSI.modulus(
            data.properties.E2,
            unitSystem
          ).toFixed(4)
        );

        setG12(
          fromSI.modulus(
            data.properties.G12,
            unitSystem
          ).toFixed(4)
        );

        setNu12(
          data.properties.nu12.toFixed(4)
        );
      } catch (err) {
        console.error(err);

        setError(
          err instanceof Error
            ? err.message
            : "Could not calculate material properties."
        );
      } finally {
        setMaterialLoading(false);
      }
    };

    calculateMaterialProperties();
  }, [fiber, matrix, Vf, unitSystem]);

  // ============================================================
  // RUN ANALYSIS
  // ============================================================

  const runAnalysis = async () => {
    setLoading(true);
    setError("");
    setResults(null);
    setStatus("Running analysis...");

    try {
      const requestBody = {
        material: {
          E1: toSI.modulus(E1, unitSystem),
          E2: toSI.modulus(E2, unitSystem),
          G12: toSI.modulus(G12, unitSystem),
          nu12: Number(nu12),
        },

        strengths: {
          Xt: toSI.strength(Xt, unitSystem),
          Xc: toSI.strength(Xc, unitSystem),
          Yt: toSI.strength(Yt, unitSystem),
          Yc: toSI.strength(Yc, unitSystem),
          S: toSI.strength(S, unitSystem),
        },

        layup,

        ply_thickness: toSI.thickness(
          plyThickness,
          unitSystem
        ),

        loads: {
          Nx: toSI.load(Nx, unitSystem),
          Ny: toSI.load(Ny, unitSystem),
          Nxy: toSI.load(Nxy, unitSystem),
          Mx: 0,
          My: 0,
          Mxy: 0,
        },
      };

      const response = await fetch(
        `${API_URL}/analyze`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(requestBody),
        }
      );

      if (!response.ok) {
        const message = await response.text();

        throw new Error(
          `API returned ${response.status}: ${message}`
        );
      }

      const data: AnalysisResponse =
        await response.json();

      setResults(data);
      setStatus("Analysis complete");
    } catch (err) {
      console.error(err);

      setStatus("Analysis failed");

      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("Unknown error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app">

      {/* ======================================================
          HEADER
          ====================================================== */}

      <header className="topbar">
        <div className="brand">
          <Layers3 size={24} />

          <div>
            <h1>CFRP Failure Prediction</h1>
            <p>
              Composite Laminate Analysis Platform
            </p>
          </div>
        </div>

        <div className="unit-selector">
          <span>Units</span>

          <select
            value={unitSystem}
            onChange={(e) =>
              handleUnitSystemChange(
                e.target.value as UnitSystem
              )
            }
          >
            <option value="metric">
              Metric
            </option>

            <option value="imperial">
              US Customary
            </option>
          </select>
        </div>

        <div className="status">
          <span
            className={`status-dot ${loading ? "status-loading" : ""
              }`}
          />

          <span>{status}</span>
        </div>
      </header>

      <main className="main-content">

        {/* ====================================================
            HERO
            ==================================================== */}

        <section className="hero">
          <div>
            <p className="eyebrow">
              ENGINEERING ANALYSIS
            </p>

            <h2>Laminate Analysis</h2>

            <p className="hero-description">
              Classical Lamination Theory and
              multi-criterion failure prediction for
              CFRP composite laminates.
            </p>
          </div>

          <Activity
            size={42}
            strokeWidth={1.5}
          />
        </section>

        {/* ====================================================
            MATERIAL SELECTION
            ==================================================== */}

        <section className="card">

          <div className="card-header">
            <div>
              <h3>Material Selection</h3>

              <p>
                Constituent materials and fiber
                volume fraction
              </p>
            </div>

            <span>Micromechanics</span>
          </div>

          <div className="form-grid">

            <label>
              Fiber Material

              <select
                value={fiber}
                onChange={(e) =>
                  setFiber(e.target.value)
                }
                disabled={
                  fibers.length === 0 ||
                  materialLoading
                }
              >
                {fibers.map((item) => (
                  <option
                    key={item}
                    value={item}
                  >
                    {item}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Matrix Material

              <select
                value={matrix}
                onChange={(e) =>
                  setMatrix(e.target.value)
                }
                disabled={
                  matrices.length === 0 ||
                  materialLoading
                }
              >
                {matrices.map((item) => (
                  <option
                    key={item}
                    value={item}
                  >
                    {item}
                  </option>
                ))}
              </select>
            </label>

            <label>
              Fiber Volume Fraction

              <div className="input-unit">
                <input
                  type="number"
                  min="0.01"
                  max="0.99"
                  step="0.01"
                  value={Vf}
                  onChange={(e) =>
                    setVf(e.target.value)
                  }
                  disabled={materialLoading}
                />

                <span>Vf</span>
              </div>
            </label>

          </div>
        </section>

        {/* ====================================================
            CALCULATED LAMINA PROPERTIES
            ==================================================== */}

        <section className="card">

          <div className="card-header">
            <div>
              <h3>
                Calculated Lamina Properties
              </h3>

              <p>
                Effective engineering constants
                calculated from the selected
                constituents
              </p>
            </div>

            <span>
              {materialLoading
                ? "Calculating..."
                : `${UNITS[unitSystem].modulus} / dimensionless`}
            </span>
          </div>

          <div className="form-grid">

            <label>
              E₁

              <div className="input-unit">
                <input
                  type="number"
                  value={E1}
                  readOnly
                />

                <span>{UNITS[unitSystem].modulus}</span>
              </div>
            </label>

            <label>
              E₂

              <div className="input-unit">
                <input
                  type="number"
                  value={E2}
                  readOnly
                />

                <span>{UNITS[unitSystem].modulus}</span>
              </div>
            </label>

            <label>
              G₁₂

              <div className="input-unit">
                <input
                  type="number"
                  value={G12}
                  readOnly
                />

                <span>{UNITS[unitSystem].modulus}</span>
              </div>
            </label>

            <label>
              ν₁₂

              <div className="input-unit">
                <input
                  type="number"
                  value={nu12}
                  readOnly
                />
              </div>
            </label>

          </div>
        </section>

        {/* ====================================================
            LAMINATE
            ==================================================== */}

        <section className="card">

          <div className="card-header">
            <h3>Laminate</h3>

            <span>Layup definition</span>
          </div>

          <label>
            Ply thickness

            <div className="input-unit">
              <input
                type="number"
                value={plyThickness}
                onChange={(e) =>
                  setPlyThickness(e.target.value)
                }
              />

              <span>{UNITS[unitSystem].thickness}</span>
            </div>
          </label>

          <div className="layup">

            <span className="layup-label">
              Layup
            </span>

            <div className="ply-editor">

              {layup.map((angle, index) => (
                <div
                  className="ply-editor-item"
                  key={index}
                >

                  <span className="ply-number">
                    Ply {index + 1}
                  </span>

                  <div className="ply-angle-input">

                    <input
                      type="number"
                      value={angle}
                      min="-180"
                      max="180"
                      step="1"
                      onChange={(e) =>
                        updatePlyAngle(
                          index,
                          e.target.value
                        )
                      }
                    />

                    <span>°</span>

                  </div>

                  <button
                    type="button"
                    className="remove-ply-button"
                    onClick={() =>
                      removePly(index)
                    }
                    disabled={
                      layup.length <= 1
                    }
                    title="Remove ply"
                  >
                    ×
                  </button>

                </div>
              ))}

            </div>

            <div className="layup-actions">

              <button
                type="button"
                className="secondary-button"
                onClick={addPly}
              >
                + Add Ply
              </button>

              <button
                type="button"
                className="secondary-button"
                onClick={resetLayup}
              >
                Reset Layup
              </button>

            </div>

          </div>

        </section>

        {/* ====================================================
            LAMINA STRENGTHS
            ==================================================== */}

        <section className="card strengths-card">

          <div className="card-header">

            <div>
              <h3>Lamina Strengths</h3>

              <p>
                Failure criterion input values
              </p>
            </div>

            <span>{UNITS[unitSystem].strength}</span>

          </div>

          <div className="strength-grid">

            <label>
              Xₜ

              <input
                type="number"
                value={Xt}
                onChange={(e) =>
                  setXt(e.target.value)
                }
              />
            </label>

            <label>
              Xc

              <input
                type="number"
                value={Xc}
                onChange={(e) =>
                  setXc(e.target.value)
                }
              />
            </label>

            <label>
              Yₜ

              <input
                type="number"
                value={Yt}
                onChange={(e) =>
                  setYt(e.target.value)
                }
              />
            </label>

            <label>
              Yc

              <input
                type="number"
                value={Yc}
                onChange={(e) =>
                  setYc(e.target.value)
                }
              />
            </label>

            <label>
              S

              <input
                type="number"
                value={S}
                onChange={(e) =>
                  setS(e.target.value)
                }
              />
            </label>

          </div>

        </section>

        {/* ====================================================
            APPLIED LOADS
            ==================================================== */}

        <section className="card loads-card">

          <div className="card-header">

            <div>
              <h3>Applied Loads</h3>

              <p>
                Membrane load resultants
              </p>
            </div>

            <span>{UNITS[unitSystem].load}</span>

          </div>

          <div className="form-grid loads">

            <label>
              Nₓ

              <input
                type="number"
                value={Nx}
                onChange={(e) =>
                  setNx(e.target.value)
                }
              />
            </label>

            <label>
              Nᵧ

              <input
                type="number"
                value={Ny}
                onChange={(e) =>
                  setNy(e.target.value)
                }
              />
            </label>

            <label>
              Nₓᵧ

              <input
                type="number"
                value={Nxy}
                onChange={(e) =>
                  setNxy(e.target.value)
                }
              />
            </label>

          </div>

          <button
            className="run-button"
            onClick={runAnalysis}
            disabled={
              loading || materialLoading
            }
          >
            <Play size={18} />

            {loading
              ? "Running Analysis..."
              : materialLoading
                ? "Calculating Material..."
                : "Run Analysis"}
          </button>

        </section>

        {/* ====================================================
            ERROR
            ==================================================== */}

        {error && (
          <section className="error-card">

            <AlertCircle size={20} />

            <div>
              <strong>
                Analysis failed
              </strong>

              <p>{error}</p>
            </div>

          </section>
        )}

        {/* ====================================================
            RESULTS
            ==================================================== */}

        {results && (
          <section className="results-section">

            <div className="results-heading">

              <div>
                <p className="eyebrow">
                  ANALYSIS RESULTS
                </p>

                <h2>
                  Failure Criterion Comparison
                </h2>
              </div>

              <div className="result-summary">
                {results.laminate.number_of_plies} plies
              </div>

            </div>

            <div className="results-grid">

              {Object.entries(
                results.governing_results
              ).map(
                ([criterion, result]) => (

                  <div
                    className="result-card"
                    key={criterion}
                  >

                    <div className="result-card-header">

                      <h3>{criterion}</h3>

                      <span
                        className={
                          result.failure_index >= 1
                            ? "failure-badge"
                            : "safe-badge"
                        }
                      >
                        {result.failure_index >= 1
                          ? "FAILURE"
                          : "SAFE"}
                      </span>

                    </div>

                    <div className="fi-value">

                      {result.failure_index.toFixed(4)}

                      <span>FI</span>

                    </div>

                    <div className="result-details">

                      <div>
                        <span>Ply</span>

                        <strong>
                          {result.ply}
                        </strong>
                      </div>

                      <div>
                        <span>Angle</span>

                        <strong>
                          {result.angle}°
                        </strong>
                      </div>

                      <div>
                        <span>Surface</span>

                        <strong>
                          {result.surface}
                        </strong>
                      </div>

                      {result.failure_mode && (
                        <div>
                          <span>Mode</span>

                          <strong>
                            {result.failure_mode}
                          </strong>
                        </div>
                      )}

                    </div>

                  </div>

                )
              )}

            </div>

          </section>
        )}

      </main>
    </div>
  );
}

export default App;