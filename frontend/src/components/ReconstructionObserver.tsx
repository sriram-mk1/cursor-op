"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, Check, Maximize2, RotateCcw, X as CloseIcon, FileText, Layers, ArrowRight, Scan, Zap, Filter, Binary } from "lucide-react";

interface Chunk {
    id: string;
    text: string;
    score: number;
    selected?: boolean;
}

export function ReconstructionObserver({ sequence }: { sequence: Chunk[] }) {
    const [phase, setPhase] = useState<"idle" | "scanning" | "analyzing" | "scoring" | "pruning" | "reordering" | "complete">("idle");
    const [statusText, setStatusText] = useState("SYSTEM READY");
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [modalTab, setModalTab] = useState<"original" | "reconstructed">("original");

    const containerRef = useRef<HTMLDivElement>(null);
    const [dimensions, setDimensions] = useState({ width: 800, height: 600 });

    // Mock sequence
    const mockSequence = Array.from({ length: 20 }).map((_, i) => ({
        id: `mock-${i}`,
        text: i % 3 === 0
            ? `System: Critical instruction set ${i} regarding security protocols and database locks...`
            : `User: Transient query regarding component padding and legacy API routes ${i}...`,
        score: i % 3 === 0 ? 0.92 : 0.15,
        selected: i % 3 === 0
    }));

    const rawSequence = (sequence && sequence.length > 0) ? sequence : mockSequence;
    const visualChunks = rawSequence.slice(0, 25);
    const filteredChunks = visualChunks.filter(c => c.selected !== undefined ? c.selected : c.score > 0.4);

    const runAnimation = async () => {
        setPhase("idle");
        setStatusText("INITIALIZING BUFFER");
        await new Promise(r => setTimeout(r, 800));

        // 1. Scanning
        setPhase("scanning");
        setStatusText("CONTEXTUAL SCAN");
        await new Promise(r => setTimeout(r, 2000));

        // 2. Analyzing
        setPhase("analyzing");
        setStatusText("RELEVANCE ANALYSIS");
        await new Promise(r => setTimeout(r, 1200));

        // 3. Scoring
        setPhase("scoring");
        setStatusText("SEMANTIC GRADING");
        await new Promise(r => setTimeout(r, 1200));

        // 4. Pruning
        setPhase("pruning");
        setStatusText("NOISE ELIMINATION");
        await new Promise(r => setTimeout(r, 1500));

        // 5. Reordering
        setPhase("reordering");
        setStatusText("STREAM RECONSTRUCTION");
        await new Promise(r => setTimeout(r, 2000));

        // 6. Complete
        setPhase("complete");
        setStatusText("OPTIMIZATION COMPLETE");
    };

    useEffect(() => {
        runAnimation();

        const handleResize = () => {
            if (containerRef.current) {
                setDimensions({
                    width: containerRef.current.clientWidth,
                    height: containerRef.current.clientHeight
                });
            }
        };

        window.addEventListener('resize', handleResize);
        handleResize();
        return () => window.removeEventListener('resize', handleResize);
    }, []);

    // Grid Props
    const COLS = 5;
    const GAP = 16;
    const CELL_W = (dimensions.width - 100 - (COLS - 1) * GAP) / COLS;
    const CELL_H = 70;

    const getGridPos = (i: number) => {
        const col = i % COLS;
        const row = Math.floor(i / COLS);
        return {
            x: 50 + col * (CELL_W + GAP),
            y: 100 + row * (CELL_H + GAP)
        };
    };

    const getReorderedPos = (id: string) => {
        const index = filteredChunks.findIndex(c => c.id === id);
        if (index === -1) return { x: 0, y: 0 };

        const REORDER_COLS = 2;
        const stackWidth = REORDER_COLS * CELL_W + (REORDER_COLS - 1) * GAP;
        const startX = (dimensions.width - stackWidth) / 2;

        const col = index % REORDER_COLS;
        const row = Math.floor(index / REORDER_COLS);

        return {
            x: startX + col * (CELL_W + GAP),
            y: 120 + row * (CELL_H + GAP)
        };
    };

    return (
        <>
            <div
                ref={containerRef}
                style={{
                    width: '100%',
                    height: '650px',
                    position: 'relative',
                    background: '#0a0a0a',
                    border: '1px solid rgba(255,255,255,0.1)',
                    marginBottom: '24px',
                    overflow: 'hidden'
                }}
            >
                {/* Background Grid */}
                <div style={{ position: 'absolute', inset: 0, opacity: 0.05, backgroundImage: 'linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)', backgroundSize: '40px 40px' }} />

                {/* Centered Top Title */}
                <div style={{ position: 'absolute', top: '24px', left: 0, right: 0, textAlign: 'center', zIndex: 20, pointerEvents: 'none' }}>
                    <span style={{ fontSize: '11px', fontWeight: 800, letterSpacing: '0.2em', color: 'rgba(255,255,255,0.3)', textTransform: 'uppercase' }}>
                        Context Migration Observer // v2.6
                    </span>
                </div>

                {/* Scanning Line with Trailing Shadow */}
                <AnimatePresence>
                    {(phase === "scanning" || phase === "analyzing") && (
                        <motion.div
                            initial={{ top: '-10%', opacity: 0 }}
                            animate={{ top: '110%', opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 3, ease: "linear" }}
                            style={{
                                position: 'absolute', left: 0, right: 0, height: '2px', background: '#00ff00', zIndex: 15, boxShadow: '0 0 10px #00ff00'
                            }}
                        >
                            <div style={{
                                position: 'absolute', bottom: '100%', left: 0, right: 0, height: '150px',
                                background: 'linear-gradient(to top, rgba(0, 255, 0, 0.2) 0%, transparent 100%)', pointerEvents: 'none'
                            }} />
                        </motion.div>
                    )}
                </AnimatePresence>

                {/* Controls */}
                <div style={{ position: 'absolute', top: '24px', right: '32px', display: 'flex', gap: '8px', zIndex: 30 }}>
                    <button onClick={runAnimation} className="hover-bright" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', width: '32px', height: '32px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fg)' }}>
                        <RotateCcw size={14} />
                    </button>
                    <button onClick={() => setIsModalOpen(true)} className="hover-bright" style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', width: '32px', height: '32px', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--fg)' }}>
                        <Maximize2 size={14} />
                    </button>
                </div>

                <AnimatePresence>
                    {visualChunks.map((chunk, i) => {
                        const isKept = chunk.selected !== undefined ? chunk.selected : chunk.score > 0.4;

                        let target = { x: 0, y: 0, opacity: 1, scale: 1, borderColor: 'rgba(255,255,255,0.1)' };
                        const initial = getGridPos(i);

                        // Standard Phase State Machine
                        switch (phase) {
                            case "idle":
                                target = { ...initial, opacity: 0, scale: 0.8, borderColor: 'rgba(255,255,255,0.1)' };
                                break;
                            case "scanning":
                                target = { ...initial, opacity: 1, scale: 1, borderColor: 'rgba(255,255,255,0.1)' };
                                break;
                            case "analyzing":
                                target = { ...initial, opacity: 1, scale: isKept ? 1.05 : 1, borderColor: isKept ? 'var(--accent)' : 'rgba(255,255,255,0.1)' };
                                break;
                            case "scoring":
                                target = { ...initial, opacity: 1, scale: 1, borderColor: isKept ? 'var(--accent)' : '#ef4444' };
                                break;
                            case "pruning":
                                target = { ...initial, opacity: isKept ? 1 : 0.05, scale: isKept ? 1 : 0.9, borderColor: isKept ? 'var(--accent)' : '#ef4444' };
                                break;
                            case "reordering":
                            case "complete":
                                if (isKept) {
                                    const reordered = getReorderedPos(chunk.id);
                                    target = { ...reordered, opacity: 1, scale: 1, borderColor: 'var(--accent)' };
                                } else {
                                    target = { ...initial, opacity: 0, scale: 0.5, borderColor: '#ef4444' };
                                }
                                break;
                        }

                        const isActive = phase === "analyzing" && isKept;

                        return (
                            <motion.div
                                key={chunk.id}
                                layoutId={`chunk-${chunk.id}`}
                                initial={phase === "idle" ? { x: initial.x, y: initial.y, opacity: 0 } : false}
                                animate={{
                                    x: target.x,
                                    y: target.y,
                                    opacity: target.opacity,
                                    scale: target.scale,
                                    borderColor: target.borderColor,
                                    backgroundColor: isActive ? 'rgba(var(--accent-rgb), 0.1)' : 'rgba(255,255,255,0.02)'
                                }}
                                transition={{ duration: 0.5, ease: "easeInOut" }}
                                style={{
                                    position: 'absolute',
                                    width: `${CELL_W}px`,
                                    height: `${CELL_H}px`,
                                    padding: '12px',
                                    border: '1px solid',
                                    color: 'var(--fg)',
                                    fontSize: '11px',
                                    fontFamily: 'monospace',
                                    display: 'flex',
                                    flexDirection: 'column',
                                    justifyContent: 'space-between',
                                    zIndex: isKept ? 10 : 1,
                                }}
                            >
                                <div style={{ opacity: 0.6, fontSize: '10px', lineHeight: '1.3', overflow: 'hidden', textOverflow: 'ellipsis', display: '-webkit-box', WebkitLineClamp: 2, WebkitBoxOrient: 'vertical' }}>
                                    {chunk.text}
                                </div>

                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '4px' }}>
                                    <div style={{ width: '4px', height: '4px', background: isKept ? 'var(--accent)' : 'rgba(255,255,255,0.1)' }} />

                                    {/* Scoring Marks */}
                                    {(phase === "scoring" || phase === "pruning" || phase === "reordering" || phase === "complete") && (
                                        <motion.div initial={{ scale: 0 }} animate={{ scale: 1 }}>
                                            {isKept ? <Check size={12} className="text-accent" /> : <X size={12} color="#ef4444" />}
                                        </motion.div>
                                    )}
                                </div>
                            </motion.div>
                        );
                    })}
                </AnimatePresence>

                {/* Bottom Right Status - NO ICONS as requested */}
                <div style={{ position: 'absolute', bottom: '32px', right: '48px', textAlign: 'right' }}>
                    <div style={{ fontSize: '32px', fontWeight: 800, color: 'rgba(255,255,255,0.15)', letterSpacing: '-0.03em', lineHeight: 1 }}>
                        {statusText}
                    </div>
                </div>
            </div>

            {/* Enhanced Detail Modal */}
            <AnimatePresence>
                {isModalOpen && (
                    <div style={{ position: 'fixed', inset: 0, zIndex: 9999, display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}>
                        <motion.div
                            initial={{ opacity: 0, scale: 0.98 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.98 }}
                            style={{
                                width: '1000px',
                                height: '85vh',
                                background: '#090909',
                                border: '1px solid rgba(255,255,255,0.1)',
                                display: 'flex',
                                flexDirection: 'column',
                                boxShadow: '0 24px 48px rgba(0,0,0,0.5)'
                            }}
                        >
                            <div style={{ padding: '32px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                                <div>
                                    <h2 style={{ fontSize: '20px', fontWeight: 700, letterSpacing: '-0.02em', color: 'white', marginBottom: '8px' }}>Context Reconstruction Report</h2>
                                    <div style={{ display: 'flex', gap: '24px', fontSize: '12px', color: 'rgba(255,255,255,0.5)' }}>
                                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}><FileText size={14} /><span>Input Nodes: {rawSequence.length}</span></div>
                                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}><ArrowRight size={14} /></div>
                                        <div style={{ display: 'flex', gap: '8px', alignItems: 'center', color: 'var(--accent)' }}><Layers size={14} /><span>Reconstructed: {filteredChunks.length}</span></div>
                                    </div>
                                </div>
                                <button onClick={() => setIsModalOpen(false)} style={{ background: 'none', border: 'none', color: 'white', opacity: 0.5, cursor: 'pointer' }}><CloseIcon size={24} /></button>
                            </div>
                            <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>
                                <div style={{ width: '240px', borderRight: '1px solid rgba(255,255,255,0.1)', background: 'rgba(255,255,255,0.02)' }}>
                                    <button onClick={() => setModalTab('original')} style={{ width: '100%', padding: '20px 24px', textAlign: 'left', background: modalTab === 'original' ? 'rgba(255,255,255,0.05)' : 'none', border: 'none', borderLeft: modalTab === 'original' ? '2px solid white' : '2px solid transparent', color: modalTab === 'original' ? 'white' : 'rgba(255,255,255,0.4)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.05em', cursor: 'pointer' }}>RAW INPUT STREAM</button>
                                    <button onClick={() => setModalTab('reconstructed')} style={{ width: '100%', padding: '20px 24px', textAlign: 'left', background: modalTab === 'reconstructed' ? 'rgba(var(--accent-rgb), 0.05)' : 'none', border: 'none', borderLeft: modalTab === 'reconstructed' ? '2px solid var(--accent)' : '2px solid transparent', color: modalTab === 'reconstructed' ? 'var(--accent)' : 'rgba(255,255,255,0.4)', fontSize: '12px', fontWeight: 700, letterSpacing: '0.05em', cursor: 'pointer' }}>OPTIMIZED CONTEXT</button>
                                </div>
                                <div style={{ flex: 1, overflowY: 'auto', background: '#050505', padding: '0' }}>
                                    <div style={{ padding: '32px', maxWidth: '800px', margin: '0 auto' }}>
                                        {(modalTab === 'original' ? rawSequence : filteredChunks).map((chunk, i) => (
                                            <div key={i} style={{ marginBottom: '16px', borderRadius: '4px', border: '1px solid rgba(255,255,255,0.05)', background: 'rgba(255,255,255,0.01)', overflow: 'hidden' }}>
                                                <div style={{ padding: '12px 20px', background: 'rgba(255,255,255,0.02)', borderBottom: '1px solid rgba(255,255,255,0.05)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                                    <div style={{ fontFamily: 'monospace', fontSize: '10px', color: 'rgba(255,255,255,0.4)' }}>
                                                        NODE_{(chunk.id || `UNKNOWN-${i}`).toUpperCase().replace(/\-/g, '_')}
                                                    </div>
                                                    <div style={{ fontSize: '10px', fontWeight: 700, color: (chunk.selected || chunk.score > 0.4) ? 'var(--accent)' : 'rgba(255,255,255,0.2)' }}>SCORE: {chunk.score.toFixed(4)}</div>
                                                </div>
                                                <div style={{ padding: '20px', fontFamily: 'monospace', fontSize: '13px', lineHeight: '1.6', color: 'rgba(255,255,255,0.7)' }}>{chunk.text}</div>
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            </div>
                        </motion.div>
                    </div>
                )}
            </AnimatePresence>
        </>
    );
}
