import { useState } from "react";
import { Activity, Layers3, Play, AlertCircle } from "lucide-react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

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

function App() {
  const [status, setStatus] = useState("Ready");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [results, setResults] = useState<AnalysisResponse | null>(null);

  // Material properties displayed in GPa
  const [E1, setE1] = useState("139.4");
  const [E2, setE2] = useState("8.5547");
  const [G12, setG12] = useState("3.0516");
  const [nu12, setNu12] = useState("0.26");

  // Strengths displayed in MPa
  const [Xt, setXt] = useState("1950");
  const [Xc, setXc] = useState("1480");
  const [Yt, setYt] = useState("48");
  const [Yc, setYc] = useState("200");
  const [S, setS] = useState("79");

  // Laminate
  const [plyThickness, setPlyThickness] = useState("0.125");

  const [layup] = useState<number[]>([
    0,
    45,
    -45,
    90,
    90,
    -45,
    45,
    0,
  ]);

  // Loads displayed in N/m
  const [Nx, setNx] = useState("0");
  const [Ny, setNy] = useState("0");
  const [Nxy, setNxy] = useState("0");

  const runAnalysis = async () => {
    setLoading(true);
    setError("");
    setResults(null);
    setStatus("Running analysis...");

    try {
      const requestBody = {
        material: {
          E1: Number(E1) * 1e9,
          E2: Number(E2) * 1e9,
          G12: Number(G12) * 1e9,
          nu12: Number(nu12),
        },

        strengths: {
          Xt: Number(Xt) * 1e6,
          Xc: Number(Xc) * 1e6,
          Yt: Number(Yt) * 1e6,
          Yc: Number(Yc) * 1e6,
          S: Number(S) * 1e6,
        },

        layup,

        ply_thickness: Number(plyThickness) / 1000,

        loads: {
          Nx: Number(Nx),
          Ny: Number(Ny),
          Nxy: Number(Nxy),
          Mx: 0,
          My: 0,
          Mxy: 0,
        },
      };

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(requestBody),
      });

      if (!response.ok) {
        const message = await response.text();
        throw new Error(
          `API returned ${response.status}: ${message}`,
        );
      }

      const data: AnalysisResponse = await response.json();

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

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <Layers3 size={24} />

          <div>
            <h1>CFRP Failure Prediction</h1>
            <p>Composite Laminate Analysis Platform</p>
          </div>
        </div>

        <div className="status">
          <span
            className={`status-dot ${
              loading ? "status-loading" : ""
            }`}
          />

          <span>{status}</span>
        </div>
      </header>

      <main className="main-content">
        <section className="hero">
          <div>
            <p className="eyebrow">ENGINEERING ANALYSIS</p>

            <h2>Laminate Analysis</h2>

            <p className="hero-description">
              Classical Lamination Theory and multi-criterion failure
              prediction for CFRP composite laminates.
            </p>
          </div>

          <Activity size={42} strokeWidth={1.5} />
        </section>

        <section className="grid">
          {/* MATERIAL */}
          <div className="card">
            <div className="card-header">
              <h3>Material Properties</h3>
              <span>GPa / dimensionless</span>
            </div>

            <div className="form-grid">
              <label>
                E₁
                <div className="input-unit">
                  <input
                    type="number"
                    value={E1}
                    onChange={(e) => setE1(e.target.value)}
                  />
                  <span>GPa</span>
                </div>
              </label>

              <label>
                E₂
                <div className="input-unit">
                  <input
                    type="number"
                    value={E2}
                    onChange={(e) => setE2(e.target.value)}
                  />
                  <span>GPa</span>
                </div>
              </label>

              <label>
                G₁₂
                <div className="input-unit">
                  <input
                    type="number"
                    value={G12}
                    onChange={(e) => setG12(e.target.value)}
                  />
                  <span>GPa</span>
                </div>
              </label>

              <label>
                ν₁₂
                <div className="input-unit">
                  <input
                    type="number"
                    value={nu12}
                    onChange={(e) => setNu12(e.target.value)}
                  />
                </div>
              </label>
            </div>
          </div>

          {/* LAMINATE */}
          <div className="card">
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
                <span>mm</span>
              </div>
            </label>

            <div className="layup">
              <span className="layup-label">Layup</span>

              <div className="ply-list">
                {layup.map((angle, index) => (
                  <div className="ply" key={index}>
                    {angle}°
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* STRENGTHS */}
        <section className="card strengths-card">
          <div className="card-header">
            <div>
              <h3>Lamina Strengths</h3>
              <p>Failure criterion input values</p>
            </div>

            <span>MPa</span>
          </div>

          <div className="strength-grid">
            <label>
              Xₜ
              <input
                type="number"
                value={Xt}
                onChange={(e) => setXt(e.target.value)}
              />
            </label>

            <label>
              X꜀
              <input
                type="number"
                value={Xc}
                onChange={(e) => setXc(e.target.value)}
              />
            </label>

            <label>
              Yₜ
              <input
                type="number"
                value={Yt}
                onChange={(e) => setYt(e.target.value)}
              />
            </label>

            <label>
              Y꜀
              <input
                type="number"
                value={Yc}
                onChange={(e) => setYc(e.target.value)}
              />
            </label>

            <label>
              S
              <input
                type="number"
                value={S}
                onChange={(e) => setS(e.target.value)}
              />
            </label>
          </div>
        </section>

        {/* LOADS */}
        <section className="card loads-card">
          <div className="card-header">
            <div>
              <h3>Applied Loads</h3>
              <p>Membrane load resultants</p>
            </div>

            <span>N/m</span>
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
          </div>

          <button
            className="run-button"
            onClick={runAnalysis}
            disabled={loading}
          >
            <Play size={18} />

            {loading ? "Running Analysis..." : "Run Analysis"}
          </button>
        </section>

        {/* ERROR */}
        {error && (
          <section className="error-card">
            <AlertCircle size={20} />

            <div>
              <strong>Analysis failed</strong>
              <p>{error}</p>
            </div>
          </section>
        )}

        {/* RESULTS */}
        {results && (
          <section className="results-section">
            <div className="results-heading">
              <div>
                <p className="eyebrow">ANALYSIS RESULTS</p>
                <h2>Failure Criterion Comparison</h2>
              </div>

              <div className="result-summary">
                {results.laminate.number_of_plies} plies
              </div>
            </div>

            <div className="results-grid">
              {Object.entries(results.governing_results).map(
                ([criterion, result]) => (
                  <div className="result-card" key={criterion}>
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
                        <strong>{result.ply}</strong>
                      </div>

                      <div>
                        <span>Angle</span>
                        <strong>{result.angle}°</strong>
                      </div>

                      <div>
                        <span>Surface</span>
                        <strong>{result.surface}</strong>
                      </div>

                      {result.failure_mode && (
                        <div>
                          <span>Mode</span>
                          <strong>{result.failure_mode}</strong>
                        </div>
                      )}
                    </div>
                  </div>
                ),
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;