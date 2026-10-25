import { useEffect, useState } from "react";
import { Activity, Layers3, Play, Sun, Moon } from "lucide-react";
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

type MaterialStrengthsResponse = {
  status: string;
  fiber: string;
  matrix: string;
  system_id?: string;
  system_name?: string;
  strengths?: {
    Xt: number;
    Xc: number;
    Yt: number;
    Yc: number;
    S: number;
  };
  source?: string;
  source_type?: string;
  notes?: string;
  message?: string;
};

type MaterialCompatibility = {
  fiber: string;
  matrix: string;
  system_id: string;
  system_name: string;
};

type MaterialsResponse = {
  fibers: string[];
  matrices: string[];
  compatible_systems: MaterialCompatibility[];
};

type UnitSystem = "metric" | "imperial";

type Theme = "dark" | "light";

function getInitialTheme(): Theme {
  try {
    const savedTheme = window.localStorage.getItem("cfrp-theme");
    return savedTheme === "light" ? "light" : "dark";
  } catch {
    return "dark";
  }
}

function App() {
  const [theme, setTheme] = useState<Theme>(getInitialTheme);
  const [status, setStatus] = useState("Ready");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    try {
      window.localStorage.setItem("cfrp-theme", theme);
    } catch {
      // Theme still works for this session if storage is unavailable.
    }
  }, [theme]);

  const toggleTheme = () => {
    setTheme((currentTheme) => currentTheme === "dark" ? "light" : "dark");
  };
  const [unitSystem, setUnitSystem] =
    useState<UnitSystem>("metric");
  const [loading, setLoading] = useState(false);
  const [materialLoading, setMaterialLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<AnalysisResponse | null>(null);
  const [mlLoading, setMlLoading] = useState(false);
  const [mlResult, setMlResult] = useState<{

    hashin_fi: number;
    failed: boolean;
    decision: string;
  } | null>(null);

  const [mlPredictions, setMlPredictions] = useState<
    Record<
      string,
      {
        name: string;
        failure_index: number;
        failed: boolean;
        decision: string;
      }
    > | null
  >(null);

  // ============================================================
  // MATERIAL SELECTION
  // ============================================================

  const [fibers, setFibers] = useState<string[]>([]);
  const [compatibleSystems, setCompatibleSystems] = useState<
    MaterialCompatibility[]
  >([]);

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
  const [strengthsAvailable, setStrengthsAvailable] = useState(false);
  const [strengthSystemName, setStrengthSystemName] = useState("");

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

  const [Mx, setMx] = useState("0");
  const [My, setMy] = useState("0");
  const [Mxy, setMxy] = useState("0");

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

    setMx(
      convertUnitValue(
        Mx,
        "moment",
        unitSystem,
        newSystem
      )
    );

    setMy(
      convertUnitValue(
        My,
        "moment",
        unitSystem,
        newSystem
      )
    );

    setMxy(
      convertUnitValue(
        Mxy,
        "moment",
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
    let cancelled = false;

    const loadMaterials = async () => {
      try {
        const response = await fetch(`${API_URL}/materials`);

        if (!response.ok) {
          throw new Error(
            `Failed to load materials: ${response.status}`
          );
        }

        const data: MaterialsResponse = await response.json();

        if (cancelled) return;

        setFibers(data.fibers);
        setCompatibleSystems(data.compatible_systems);

        const initialSystem =
          data.compatible_systems.find(
            (system) =>
              system.fiber === "T300" &&
              system.matrix === "Standard Epoxy"
          ) ?? data.compatible_systems[0];

        if (initialSystem) {
          setFiber(initialSystem.fiber);
          setMatrix(initialSystem.matrix);
        } else {
          setStrengthsAvailable(false);
          setStrengthSystemName("");
          setError(
            "No supported material-strength systems are configured."
          );
        }
      } catch (err) {
        if (cancelled) return;

        console.error(err);
        setError(
          "Could not load the material database. Make sure the FastAPI server is running."
        );
      }
    };

    loadMaterials();

    return () => {
      cancelled = true;
    };
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
      setStrengthsAvailable(false);
      setStrengthSystemName("");
      setXt("");
      setXc("");
      setYt("");
      setYc("");
      setS("");

      try {
        // ==========================================================
        // EFFECTIVE LAMINA PROPERTIES
        // ==========================================================

        const materialResponse = await fetch(
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

        if (!materialResponse.ok) {
          const message = await materialResponse.text();

          throw new Error(
            `Material API returned ${materialResponse.status}: ${message}`
          );
        }

        const materialData: MaterialPropertiesResponse =
          await materialResponse.json();

        if (materialData.status !== "success") {
          throw new Error(
            "Material-property calculation failed."
          );
        }

        setE1(
          fromSI.modulus(
            materialData.properties.E1,
            unitSystem
          ).toFixed(4)
        );

        setE2(
          fromSI.modulus(
            materialData.properties.E2,
            unitSystem
          ).toFixed(4)
        );

        setG12(
          fromSI.modulus(
            materialData.properties.G12,
            unitSystem
          ).toFixed(4)
        );

        setNu12(
          materialData.properties.nu12.toFixed(4)
        );

        // ==========================================================
        // VERIFIED LAMINA STRENGTHS
        // ==========================================================

        const strengthResponse = await fetch(
          `${API_URL}/material-strengths?fiber=${encodeURIComponent(
            fiber
          )}&matrix=${encodeURIComponent(matrix)}`
        );

        if (!strengthResponse.ok) {
          const message = await strengthResponse.text();

          throw new Error(
            `Strength API returned ${strengthResponse.status}: ${message}`
          );
        }

        const strengthData: MaterialStrengthsResponse =
          await strengthResponse.json();

        if (
          strengthData.status !== "success" ||
          !strengthData.strengths
        ) {
          setError(
            strengthData.message ||
            `No verified strength dataset is available for ${fiber}/${matrix}.`
          );
          return;
        }

        setXt(
          fromSI.strength(
            strengthData.strengths.Xt,
            unitSystem
          ).toFixed(4)
        );

        setXc(
          fromSI.strength(
            strengthData.strengths.Xc,
            unitSystem
          ).toFixed(4)
        );

        setYt(
          fromSI.strength(
            strengthData.strengths.Yt,
            unitSystem
          ).toFixed(4)
        );

        setYc(
          fromSI.strength(
            strengthData.strengths.Yc,
            unitSystem
          ).toFixed(4)
        );

        setS(
          fromSI.strength(
            strengthData.strengths.S,
            unitSystem
          ).toFixed(4)
        );


        setStrengthsAvailable(true);
        setStrengthSystemName(
          strengthData.system_name || strengthData.system_id || ""
        );
        setError("");

      } catch (err) {
        console.error(err);

        setError(
          err instanceof Error
            ? err.message
            : "Failed to load material properties and strengths."
        );
      } finally {
        setMaterialLoading(false);
      }
    };

    calculateMaterialProperties();
  }, [fiber, matrix, Vf, unitSystem]);

  // ============================================================
  // INPUT VALIDATION
  // ============================================================

  const validateInputs = (): string | null => {
    const numericFields: Array<[string, string]> = [
      ["Fiber volume fraction", Vf],
      ["Ply thickness", plyThickness],
      ["E₁", E1],
      ["E₂", E2],
      ["G₁₂", G12],
      ["ν₁₂", nu12],
      ["Longitudinal tensile strength (Xt)", Xt],
      ["Longitudinal compressive strength (Xc)", Xc],
      ["Transverse tensile strength (Yt)", Yt],
      ["Transverse compressive strength (Yc)", Yc],
      ["Shear strength (S)", S],
      ["Nx", Nx],
      ["Ny", Ny],
      ["Nxy", Nxy],
      ["Mx", Mx],
      ["My", My],
      ["Mxy", Mxy],
    ];

    for (const [label, rawValue] of numericFields) {
      if (rawValue.trim() === "" || !Number.isFinite(Number(rawValue))) {
        return `${label} must contain a valid number.`;
      }
    }

    const vf = Number(Vf);
    if (vf <= 0 || vf >= 1) {
      return "Fiber volume fraction must be greater than 0 and less than 1 (for example, 0.60).";
    }

    if (Number(plyThickness) <= 0) {
      return "Ply thickness must be greater than 0.";
    }

    if (layup.length < 1) {
      return "The laminate must contain at least one ply.";
    }

    if (layup.some((angle) => !Number.isFinite(angle) || angle < -180 || angle > 180)) {
      return "Every ply angle must be between -180° and 180°.";
    }

    if ([Number(E1), Number(E2), Number(G12)].some((value) => value <= 0)) {
      return "Elastic moduli E₁, E₂, and G₁₂ must be greater than 0.";
    }

    const poisson = Number(nu12);
    if (poisson <= 0 || poisson >= 0.5) {
      return "Poisson's ratio ν₁₂ must be greater than 0 and less than 0.5.";
    }

    if ([Number(Xt), Number(Xc), Number(Yt), Number(Yc), Number(S)].some((value) => value <= 0)) {
      return "All strength values must be greater than 0.";
    }

    return null;
  };

  // ============================================================
  // RUN ANALYSIS
  // ============================================================

  const runAnalysis = async () => {
    const validationError = validateInputs();
    if (validationError) {
      setError(validationError);
      setStatus("Check inputs");
      setResults(null);
      return;
    }

    if (!strengthsAvailable) {
      setError(
        "Failure analysis is unavailable because this material combination has no verified strength dataset."
      );
      return;
    }
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
          Mx: toSI.moment(Mx, unitSystem),
          My: toSI.moment(My, unitSystem),
          Mxy: toSI.moment(Mxy, unitSystem),
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

  const runMLPrediction = async () => {
    const validationError = validateInputs();
    if (validationError) {
      setError(validationError);
      setStatus("Check inputs");
      setMlResult(null);
      setMlPredictions(null);
      return;
    }

    setMlLoading(true);
    setError("");

    try {
      const requestBody = {
        Nx: toSI.load(Nx, unitSystem),
        Ny: toSI.load(Ny, unitSystem),
        Nxy: toSI.load(Nxy, unitSystem),
        Mx: toSI.moment(Mx, unitSystem),
        My: toSI.moment(My, unitSystem),
        Mxy: toSI.moment(Mxy, unitSystem),
      };

      const response = await fetch(
        `${API_URL}/ml-predict`,
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
          `ML API returned ${response.status}: ${message}`
        );
      }

      const data = await response.json();

      setMlResult(data.prediction);
      setMlPredictions(data.predictions);
    } catch (err) {
      console.error(err);

      if (err instanceof Error) {
        setError(err.message);
      } else {
        setError("ML prediction failed.");
      }
    } finally {
      setMlLoading(false);
    }
  };

  // ============================================================
  // UI
  // ============================================================
  const compatibleMatrices = compatibleSystems
    .filter((system) => system.fiber === fiber)
    .map((system) => system.matrix);


  return (
    <div className="app">
      {error && (
        <div
          role="alert"
          className="error-message"
          style={{
            padding: "12px 16px",
            marginBottom: "16px",
            border: "1px solid #dc2626",
            borderRadius: "8px",
            color: "#dc2626",
          }}
        >
          {error}
        </div>
      )}

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

        <button
          type="button"
          className="theme-toggle"
          onClick={toggleTheme}
          aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
          title={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
        >
          {theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}
          <span>{theme === "dark" ? "Light mode" : "Dark mode"}</span>
        </button>

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
                onChange={(e) => {
                  const nextFiber = e.target.value;
                  const nextSystem = compatibleSystems.find(
                    (system) => system.fiber === nextFiber
                  );

                  setFiber(nextFiber);
                  setStrengthsAvailable(false);
                  setStrengthSystemName("");

                  if (nextSystem) {
                    setMatrix(nextSystem.matrix);
                  } else {
                    setError(
                      `No verified strength dataset is configured for ${nextFiber}.`
                    );
                  }
                }}
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
                onChange={(e) => {
                  setMatrix(e.target.value);
                  setStrengthsAvailable(false);
                  setStrengthSystemName("");
                }}
                disabled={
                  compatibleMatrices.length === 0 ||
                  materialLoading
                }
              >
                {compatibleMatrices.map((item) => (
                  <option key={item} value={item}>
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
                  aria-invalid={Vf.trim() === "" || !Number.isFinite(Number(Vf)) || Number(Vf) <= 0 || Number(Vf) >= 1}
                  aria-describedby="vf-help"
                  onChange={(e) => {
                    const nextValue = e.target.value;
                    setVf(nextValue);
                    const nextVf = Number(nextValue);
                    if (nextValue.trim() === "" || !Number.isFinite(nextVf) || nextVf <= 0 || nextVf >= 1) {
                      setError("Fiber volume fraction must be greater than 0 and less than 1 (for example, 0.60).");
                    } else {
                      setError("");
                    }
                  }}
                  disabled={materialLoading}
                />

                <span>Vf</span>
              </div>
              <span id="vf-help" className="ml-note">
                Enter a value greater than 0 and less than 1 (e.g. 0.60).
              </span>
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

              {strengthsAvailable ? (
                <p className="strength-source-status">
                  Verified dataset: {strengthSystemName}
                </p>
              ) : (
                <p className="strength-source-status strength-unavailable">
                  No verified strength dataset loaded for this selection.
                </p>
              )}

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

            <span>
              N: {UNITS[unitSystem].load} · M: {UNITS[unitSystem].moment}
            </span>

          </div>

          <div className="form-grid loads">
            <label>
              Nₓ
              <input
                type="number"
                value={Nx}
                onChange={(e) => setNx(e.target.value)}
              />
            </label>

            <label>
              Nᵧ
              <input
                type="number"
                value={Ny}
                onChange={(e) => setNy(e.target.value)}
              />
            </label>

            <label>
              Nₓᵧ
              <input
                type="number"
                value={Nxy}
                onChange={(e) => setNxy(e.target.value)}
              />
            </label>

            <label>
              Mₓ
              <input
                type="number"
                value={Mx}
                onChange={(e) => setMx(e.target.value)}
              />
            </label>

            <label>
              Mᵧ
              <input
                type="number"
                value={My}
                onChange={(e) => setMy(e.target.value)}
              />
            </label>

            <label>
              Mₓᵧ
              <input
                type="number"
                value={Mxy}
                onChange={(e) => setMxy(e.target.value)}
              />
            </label>
          </div>

          <button
            className="run-button"
            onClick={runAnalysis}
            disabled={
              loading ||
              materialLoading ||
              !strengthsAvailable
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


        {results && (
          <section className="card">
            <div className="card-header">
              <div>
                <h3>Laminate Analysis Results</h3>
                <p>Classical Laminate Theory and failure criteria</p>
              </div>
              <span>{results.laminate.number_of_plies} plies</span>
            </div>

            <h4>Governing Failure Indices</h4>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                gap: "12px",
              }}
            >
              {Object.entries(results.governing_results).map(([name, result]) => (
                <div
                  key={name}
                  style={{
                    padding: "16px",
                    border: "1px solid var(--border-color, #444)",
                    borderRadius: "10px",
                  }}
                >
                  <strong>{name}</strong>
                  <div className="fi-value">
                    {result.failure_index.toFixed(4)}
                    <span> FI</span>
                  </div>
                  <span
                    className={
                      result.failure_index >= 1
                        ? "failure-badge"
                        : "safe-badge"
                    }
                  >
                    {result.failure_index >= 1 ? "FAILURE" : "SAFE"}
                  </span>
                  <p className="ml-note">
                    Governing ply: {result.ply} · {result.surface}
                  </p>
                </div>
              ))}
            </div>

            <h4 style={{ marginTop: "24px" }}>Laminate Response</h4>
            <p>
              Mid-plane strain:{" "}
              {results.response.mid_plane_strain
                .map((value) => value.toExponential(3))
                .join(", ")}
            </p>
            <p>
              Curvature:{" "}
              {results.response.curvature
                .map((value) => value.toExponential(3))
                .join(", ")}
            </p>
            <p className="ml-note">
              Failure is classified using FI ≥ 1.0. These are computational
              predictions, not experimental validation.
            </p>
          </section>
        )}


        <section className="card ml-card">
          <div className="card-header">
            <div>
              <h3>AI Failure Prediction</h3>
              <p>
                DNN surrogate prediction of the Hashin failure index
              </p>
            </div>

            <span>DNN • [64, 64, 32]</span>
          </div>

          <button
            className="run-button"
            onClick={runMLPrediction}
            disabled={mlLoading || materialLoading}
          >
            {mlLoading
              ? "Predicting..."
              : "Predict Failure with AI"}
          </button>


          {mlResult && (
            <div className="ml-result">
              <div className="ml-result-header">
                <span>AI Failure Prediction Results</span>
                <span
                  className={
                    mlResult.failed ? "failure-badge" : "safe-badge"
                  }
                >
                  Hashin: {mlResult.decision}
                </span>
              </div>

              <p className="ml-note">
                Seven failure criteria · Failure threshold: FI = 1.0
              </p>

              {mlPredictions && (
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))",
                    gap: "12px",
                    marginTop: "16px",
                  }}
                >
                  {Object.entries(mlPredictions).map(([key, result]) => (
                    <div
                      key={key}
                      style={{
                        padding: "16px",
                        border: "1px solid var(--border-color, #444)",
                        borderRadius: "10px",
                        minWidth: 0,
                      }}
                    >
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          gap: "8px",
                          marginBottom: "12px",
                        }}
                      >
                        <strong>{result.name}</strong>
                        <span
                          className={
                            result.failed ? "failure-badge" : "safe-badge"
                          }
                        >
                          {result.decision}
                        </span>
                      </div>

                      <div className="fi-value">
                        {result.failure_index.toFixed(4)}
                        <span> FI</span>
                      </div>

                      <p className="ml-note">
                        {result.failed
                          ? "Failure index ≥ 1.0"
                          : "Failure index < 1.0"}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              <p className="ml-note" style={{ marginTop: "16px" }}>
                These values are predictions from trained surrogate models.
                Experimental validation is outside this project's scope.
              </p>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

export default App;