"use client";

import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity, Zap, Shield, Database, Clock, Cpu,
  ArrowUpRight, TrendingUp, Layers, ZapOff
} from "lucide-react";
import { Sidebar } from "@/components/Sidebar";
import { createClient } from "@/utils/supabase";

interface RequestLog {
  id: string;
  session_id: string;
  model: string;
  tokens_in: number;
  tokens_out: number;
  tokens_saved: number;
  latency_ms: number;
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
      if (!key || key === "v1-test-key") {
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

      setApiKey(key || "v1-test-key");
    };
    init();
  }, []);

  useEffect(() => {
    if (!apiKey) return;

    const connect = () => {
      const socket = new WebSocket(`${WS_URL}/ws/${apiKey}`);
      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        const message = JSON.parse(event.data);
        if (message.type === "init") {
          setStats(message.data);
          setLogs(message.data.recent_requests || []);
        } else if (message.type === "request") {
          const newLog = message.data;
          setLogs((prev) => [newLog, ...prev].slice(0, 50));
          setStats((prev) => prev ? {
            ...prev,
            total_tokens_saved: prev.total_tokens_saved + newLog.tokens_saved,
            total_requests: prev.total_requests + 1
          } : null);
        }
      };
      socket.onclose = () => {
        setConnected(false);
        setTimeout(connect, 3000);
      };
      ws.current = socket;
    };
    connect();
    return () => ws.current?.close();
  }, [apiKey]);

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
          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="glass card stat-item">
            <div className="stat-label"><Zap size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Tokens Saved</div>
            <div className="stat-value text-accent">{(stats?.total_tokens_saved || 0).toLocaleString()}</div>
            <div style={{ fontSize: "11px", color: "rgba(0,255,136,0.5)", display: "flex", alignItems: "center", gap: "4px" }}>
              <TrendingUp size={12} /> Real-time optimization
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass card stat-item">
            <div className="stat-label"><Activity size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Total Requests</div>
            <div className="stat-value">{(stats?.total_requests || 0).toLocaleString()}</div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>Across {new Set(logs.map(l => l.session_id)).size} active sessions</div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass card stat-item">
            <div className="stat-label"><Shield size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Avg. Latency</div>
            <div className="stat-value">
              {logs.length > 0
                ? (logs.reduce((acc, l) => acc + l.latency_ms, 0) / logs.length).toFixed(0)
                : "0"}
              <span style={{ fontSize: "14px", opacity: 0.4, marginLeft: "4px" }}>ms</span>
            </div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>
              P95: {logs.length > 0 ? [...logs].sort((a, b) => b.latency_ms - a.latency_ms)[Math.floor(logs.length * 0.05)]?.latency_ms.toFixed(0) : "0"}ms
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="glass card stat-item">
            <div className="stat-label"><Layers size={12} style={{ verticalAlign: "middle", marginRight: "6px" }} /> Optimization Rate</div>
            <div className="stat-value">
              {logs.length > 0
                ? ((logs.reduce((acc, l) => acc + l.tokens_saved, 0) / (logs.reduce((acc, l) => acc + l.tokens_in, 0) || 1)) * 100).toFixed(1)
                : "0.0"}%
            </div>
            <div style={{ fontSize: "11px", opacity: 0.4 }}>V1 Context Savings</div>
          </motion.div>
        </div>

        {/* Live Feed Section */}
        <section className="glass" style={{ padding: "0", overflow: "hidden" }}>
          <div className="card-header" style={{ padding: "20px 24px", borderBottom: "1px solid var(--border)" }}>
            <h2 style={{ fontSize: "14px", fontWeight: 600 }}>Live Request Stream</h2>
            <div className="btn btn-outline" style={{ fontSize: "10px", padding: "4px 10px" }}>
              Clear Logs
            </div>
          </div>

          <div className="log-list">
            <div className="log-item log-header">
              <div>Session ID</div>
              <div>Model</div>
              <div>Savings</div>
              <div>Latency</div>
              <div style={{ textAlign: "right" }}>Time</div>
            </div>

            <AnimatePresence initial={false}>
              {logs.map((log) => (
                <motion.div
                  key={log.id}
                  initial={{ opacity: 0, height: 0 }}
                  animate={{ opacity: 1, height: "auto" }}
                  exit={{ opacity: 0 }}
                  className="log-item"
                >
                  <div style={{ display: "flex", alignItems: "center", gap: "10px" }}>
                    <div style={{ width: "4px", height: "4px", background: "var(--accent)", borderRadius: "50%" }} />
                    <span style={{ opacity: 0.9 }}>{log.session_id}</span>
                  </div>
                  <div className="text-muted" style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <Cpu size={12} /> {log.model}
                  </div>
                  <div style={{ color: log.tokens_saved > 0 ? "var(--accent)" : "rgba(255,255,255,0.2)" }}>
                    {log.tokens_saved > 0 ? (
                      <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                        <Zap size={12} /> {log.tokens_saved.toLocaleString()}
                      </span>
                    ) : (
                      <span style={{ display: "flex", alignItems: "center", gap: "4px" }}>
                        <ZapOff size={12} /> 0
                      </span>
                    )}
                  </div>
                  <div className="text-muted">{log.latency_ms.toFixed(0)}ms</div>
                  <div style={{ textAlign: "right", opacity: 0.4 }}>
                    {new Date(log.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </div>
                </motion.div>
              ))}
            </AnimatePresence>

            {logs.length === 0 && (
              <div style={{ padding: "60px", textAlign: "center", background: "var(--bg)" }}>
                <Activity size={32} className="text-muted" style={{ marginBottom: "16px", opacity: 0.2 }} />
                <p className="text-muted" style={{ fontSize: "13px" }}>Listening for incoming requests on <code style={{ color: "var(--accent)" }}>:8000</code>...</p>
              </div>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
