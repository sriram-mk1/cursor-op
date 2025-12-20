"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity, Zap, Shield, Database, Clock, Cpu,
  ArrowUpRight, TrendingUp, Layers, ZapOff
} from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { createClient } from "@/utils/supabase";

interface ReconstructionLog {
  total_lines: number;
  selected_lines: number;
  overhead_ms: number;
  sequence: Array<{
    source: string;
    text: string;
    score: number;
    line_index: number;
  }>;
}

interface RequestLog {
  id: string;
  session_id: string;
  model: string;
  tokens_in: number;
  tokens_out: number;
  tokens_saved: number;
  potential_total?: number; // V4 Metric: Total unoptimized history
  latency_ms: number;
  cost_saved_usd?: number;
  total_cost_usd?: number;
  reconstruction_log?: ReconstructionLog;
  timestamp: number;
}

interface Stats {
  total_tokens_saved: number;
  total_requests: number;
  recent_requests: RequestLog[];
}

export default function Dashboard() {
  const [stats, setStats] = useState<Stats | null>(null);
  const [logs, setLogs] = useState<RequestLog[]>([]);
  const [connected, setConnected] = useState(false);
  const [expandedLogId, setExpandedLogId] = useState<string | null>(null);
  const [visibleChunks, setVisibleChunks] = useState<Record<string, number>>({});
  const ws = useRef<WebSocket | null>(null);

  const [apiKey, setApiKey] = useState<string>("");
  const GATEWAY_URL = process.env.NEXT_PUBLIC_GATEWAY_URL || "http://localhost:8000";
  const WS_URL = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";
  const supabase = createClient();

  useEffect(() => {
    const init = async () => {
      // 1. Try to get key from localStorage
      let key = typeof window !== 'undefined' ? localStorage.getItem("v1_key") : null;

      // 2. If no key, fetch from Supabase
      if (!key) {
        const { data: { user } } = await supabase.auth.getUser();
        if (user) {
          try {
            const resp = await fetch(`${GATEWAY_URL}/api/keys`, {
              headers: { "x-user-id": user.id }
            });
            if (resp.ok) {
              const keys = await resp.json();
              if (Array.isArray(keys) && keys.length > 0) {
                // Use the most recently created key
                key = keys[0].raw_key;
                localStorage.setItem("v1_key", key!);
              }
            }
          } catch (e) {
            console.error("Failed to fetch keys:", e);
          }
        }
      }

      setApiKey(key || "");
    };
    init();
  }, []);

  useEffect(() => {
    if (!apiKey) return;

    let socket: WebSocket | null = null;
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      if (socket?.readyState === WebSocket.OPEN) return;

      console.log(`Connecting to WS with key: ${apiKey}`);
      socket = new WebSocket(`${WS_URL}/ws/${apiKey}`);

      socket.onopen = () => {
        console.log("WS Connected");
        setConnected(true);
      };

      socket.onmessage = async (event) => {
        try {
          const message = JSON.parse(event.data);
          console.log("WS Message:", message.type);

          if (message.type === "init") {
            // After WS init, we override with aggregate user stats for a "synced" experience
            const { data: { user } } = await supabase.auth.getUser();
            if (user) {
              const resp = await fetch(`${GATEWAY_URL}/api/user/stats`, {
                headers: { "x-user-id": user.id }
              });
              const data = await resp.json();
              setStats(data);
              setLogs(data.recent_requests || []);
            }
          } else if (message.type === "request") {
            const newLog = message.data;
            setLogs((prev) => {
              const index = prev.findIndex(l => l.id === newLog.id);
              if (index !== -1) {
                // Update existing
                const updated = [...prev];
                updated[index] = newLog;
                return updated;
              }
              return [newLog, ...prev].slice(0, 50);
            });
            setStats((prev) => prev ? {
              ...prev,
              total_tokens_saved: prev.total_tokens_saved + newLog.tokens_saved,
              total_requests: prev.total_requests + 1
            } : null);
          }
        } catch (e) {
          console.error("WS Parse Error:", e);
        }
      };

      socket.onclose = (e) => {
        console.log("WS Closed:", e.code, e.reason);
        setConnected(false);
        // Only reconnect if not 4003 (invalid key)
        if (e.code !== 4003) {
          reconnectTimeout = setTimeout(connect, 3000);
        }
      };

      socket.onerror = (e) => {
        console.error("WS Error:", e);
      };

      ws.current = socket;
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (socket) socket.close();
    };
  }, [apiKey]);

  const totalCostSaved = logs.reduce((acc, log) => acc + (log.cost_saved_usd || 0), 0);

  return (
    <div className="layout-wrapper">
      <Sidebar />

      <main className="main-content">
        {/* Top Header */}
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "48px" }}>
          <div>
            <h1 style={{ fontSize: "24px", marginBottom: "4px" }}>Overview</h1>
            <p className="text-muted" style={{ fontSize: "14px" }}>Real-time performance and cost optimization metrics.</p>
          </div>
          <div className="glass" style={{ padding: "8px 16px", display: "flex", alignItems: "center", gap: "12px" }}>
            <div className={connected ? "live-indicator" : ""} style={{ background: connected ? "var(--accent)" : "#ff4444" }} />
            <span style={{ fontSize: "11px", fontWeight: 600 }}>{connected ? "GATEWAY ONLINE" : "GATEWAY OFFLINE"}</span>
          </div>
        </header>

        {/* Stats Grid */}
        <div className="stats-grid">
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass card stat-item" style={{ background: "var(--accent-muted)", borderColor: "var(--accent)" }}>
            <div className="stat-label"><Zap size={12} style={{ verticalAlign: "middle", marginRight: "6px", color: "var(--accent)" }} /> Est. Cost Saved</div>
            <div className="stat-value text-accent" style={{ fontSize: "32px" }}>${totalCostSaved.toFixed(4)}</div>
            <div style={{ fontSize: "11px", color: "rgba(0,255,136,0.6)", display: "flex", alignItems: "center", gap: "4px" }}>
              <TrendingUp size={12} /> Optimization Yield
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.1 }} className="glass card stat-item">
            <div className="stat-label"><Database size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Monthly Projection</div>
            <div className="stat-value" style={{ fontSize: "32px" }}>${(totalCostSaved * 30).toFixed(2)}</div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>Estimated 30-day run rate</div>
          </motion.div>

          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.2 }} className="glass card stat-item">
            <div className="stat-label"><Shield size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Optimization Rate</div>
            <div className="stat-value" style={{ fontSize: "32px" }}>
              {logs.length > 0
                ? ((logs.reduce((acc, l) => acc + l.tokens_saved, 0) / (logs.reduce((acc, l) => acc + (l.potential_total || (l.tokens_in + l.tokens_saved)), 0) || 1)) * 100).toFixed(1)
                : "0.0"}%
            </div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>V1 Engine Efficiency</div>
          </motion.div>

          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: 0.3 }} className="glass card stat-item" style={{ borderLeft: "4px solid var(--accent)" }}>
            <div className="stat-label"><Activity size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Engine Health</div>
            <div className="stat-value" style={{ fontSize: "32px" }}>100%</div>
            <div style={{ fontSize: "11px", color: "var(--accent)" }}>Optimal Retrieval State</div>
          </motion.div>
        </div>

        {/* Home Page Enrichment: Capabilities Section */}
        <section style={{ marginBottom: "48px" }}>
          <h2 style={{ fontSize: "14px", fontWeight: 700, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.1em", marginBottom: "20px" }}>Engine Capabilities</h2>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "20px" }}>
            <CapabilityCard
              icon={<Shield size={18} className="text-accent" />}
              title="Instruction Safe"
              desc="System prompts and tool definitions are surgically preserved, ensuring reliability."
            />
            <CapabilityCard
              icon={<Layers size={18} style={{ color: "var(--purple)" }} />}
              title="Multi-Session RAG"
              desc="Context is pulled from entire conversation history across multiple sessions."
            />
            <CapabilityCard
              icon={<Zap size={18} style={{ color: "var(--yellow)" }} />}
              title="Cost Arbitrage"
              desc="Intelligently scales context based on token price and model capability."
            />
          </div>
        </section>

        <div style={{ display: "grid", gridTemplateColumns: "1.2fr 1.8fr", gap: "32px" }}>
          {/* LEFT: LIVE STREAM */}
          <section>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <h2 style={{ fontSize: "14px", fontWeight: 600, display: "flex", alignItems: "center", gap: "8px" }}>
                <div className="live-indicator" style={{ width: "8px", height: "8px" }} />
                Live Request Stream
              </h2>
              <span className="text-muted" style={{ fontSize: "11px" }}>Real-time updates</span>
            </div>

            <div className="glass" style={{ height: "600px", overflowY: "auto", border: "1px solid var(--border)", background: "rgba(0,0,0,0.2)" }}>
              <AnimatePresence initial={false}>
                {logs.length === 0 ? (
                  <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", opacity: 0.3 }}>
                    Waiting for requests...
                  </div>
                ) : logs.slice(0, 50).map((log) => (
                  <LiveLogItem key={log.id} log={log} />
                ))}
              </AnimatePresence>
            </div>
          </section>

          {/* RIGHT: HISTORICAL PERFORMANCE */}
          <section>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "16px" }}>
              <h2 style={{ fontSize: "14px", fontWeight: 600, display: "flex", alignItems: "center", gap: "8px" }}>
                <Clock size={16} className="text-muted" />
                Historical Performance
              </h2>
              <span className="text-muted" style={{ fontSize: "11px" }}>Session Archive</span>
            </div>
            <HistoryTable logs={logs} />
          </section>
        </div>
      </main>

    </div>
  );
}

function CapabilityCard({ icon, title, desc }: any) {
  return (
    <div className="glass card" style={{ padding: "24px" }}>
      <div style={{ marginBottom: "16px" }}>{icon}</div>
      <h3 style={{ fontSize: "14px", fontWeight: 600, marginBottom: "8px" }}>{title}</h3>
      <p style={{ fontSize: "12px", color: "var(--muted)", lineHeight: "1.5" }}>{desc}</p>
    </div>
  );
}

function LiveLogItem({ log }: { log: RequestLog }) {
  return (
    <motion.div
      initial={{ opacity: 0, x: -10 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0 }}
      style={{ padding: "16px 20px", borderBottom: "1px solid var(--border)", display: "grid", gridTemplateColumns: "1fr 1fr 100px", alignItems: "center", gap: "10px", cursor: "pointer" }}
      onClick={() => window.location.href = `/analytics/${log.id}`}
    >
      <div style={{ overflow: "hidden" }}>
        <div style={{ fontSize: "12px", fontWeight: 600, color: "var(--fg)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
          {log.session_id}
        </div>
        <div style={{ fontSize: "10px", opacity: 0.4 }}>{log.model.split('/')[1] || log.model}</div>
      </div>
      <div>
        <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--accent)", fontSize: "11px", fontWeight: 700 }}>
          <Zap size={10} /> {log.tokens_saved.toLocaleString()} SAVED
        </div>
        <div style={{ fontSize: "10px", opacity: 0.4 }}>${(log.cost_saved_usd || 0).toFixed(5)} opt</div>
      </div>
      <div style={{ textAlign: "right", fontSize: "10px", opacity: 0.3, fontFamily: "monospace" }}>
        {new Date(log.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
      </div>
    </motion.div>
  );
}

function HistoryTable({ logs }: { logs: RequestLog[] }) {
  const [page, setPage] = useState(0);
  const perPage = 10;
  const history = logs.slice(page * perPage, (page + 1) * perPage);

  return (
    <div className="glass card" style={{ padding: "0", height: "600px", display: "flex", flexDirection: "column" }}>
      <div style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h2 style={{ fontSize: "13px", fontWeight: 600 }}>Log History</h2>
        <div style={{ display: "flex", gap: "8px" }}>
          <button
            disabled={page === 0}
            onClick={() => setPage(p => p - 1)}
            className="btn btn-outline"
            style={{ fontSize: "10px", padding: "4px 8px", opacity: page === 0 ? 0.3 : 1 }}
          >
            Prev
          </button>
          <button
            disabled={logs.length < (page + 1) * perPage}
            onClick={() => setPage(p => p + 1)}
            className="btn btn-outline"
            style={{ fontSize: "10px", padding: "4px 8px", opacity: logs.length < (page + 1) * perPage ? 0.3 : 1 }}
          >
            Next
          </button>
        </div>
      </div>

      <div style={{ flex: 1, overflowY: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <thead>
            <tr style={{ borderBottom: "1px solid var(--border)", background: "rgba(255,255,255,0.01)" }}>
              <th style={{ textAlign: "left", padding: "12px 24px", fontSize: "10px", fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>Request</th>
              <th style={{ textAlign: "right", padding: "12px 24px", fontSize: "10px", fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>Efficiency</th>
              <th style={{ textAlign: "right", padding: "12px 24px", fontSize: "10px", fontWeight: 700, color: "var(--muted)", textTransform: "uppercase" }}>Saved</th>
            </tr>
          </thead>
          <tbody>
            {history.map(log => (
              <tr key={log.id} style={{ borderBottom: "1px solid var(--border)", cursor: "pointer" }} onClick={() => window.location.href = `/analytics/${log.id}`}>
                <td style={{ padding: "16px 24px" }}>
                  <div style={{ fontSize: "12px", fontWeight: 500 }}>{log.id.slice(0, 12)}...</div>
                  <div style={{ fontSize: "10px", opacity: 0.4 }}>{log.model.split('/')[1] || log.model}</div>
                </td>
                <td style={{ padding: "16px 24px", textAlign: "right" }}>
                  <div style={{ fontSize: "12px", color: "var(--accent)", fontWeight: 700 }}>
                    {((log.tokens_saved / (log.potential_total || (log.tokens_in + log.tokens_saved) || 1)) * 100).toFixed(0)}%
                  </div>
                </td>
                <td style={{ padding: "16px 24px", textAlign: "right" }}>
                  <div style={{ fontSize: "12px" }}>${(log.cost_saved_usd || 0).toFixed(4)}</div>
                  <div style={{ fontSize: "10px", opacity: 0.4 }}>{log.tokens_saved.toLocaleString()} tokens</div>
                </td>
              </tr>
            ))}
            {history.length === 0 && (
              <tr>
                <td colSpan={3} style={{ padding: "40px", textAlign: "center", opacity: 0.3, fontSize: "12px" }}>No history recorded</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
