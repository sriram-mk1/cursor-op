"use client";

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function Box({ position, delay }: { position: [number, number, number], delay: number }) {
    const meshRef = useRef<THREE.Mesh>(null);

    useFrame((state) => {
        if (meshRef.current) {
            const t = state.clock.getElapsedTime() + delay;
            meshRef.current.position.y = position[1] + Math.sin(t * 2) * 0.2;
            meshRef.current.rotation.x = t * 0.5;
            meshRef.current.rotation.y = t * 0.3;
        }
    });

    return (
        <mesh ref={meshRef} position={position}>
            <boxGeometry args={[0.6, 0.6, 0.6]} />
            <meshStandardMaterial
                color="#ffffff"
                wireframe={true}
                transparent={true}
                opacity={0.15}
            />
        </mesh>
    );
}

export function HeroCubes() {
    return (
        <div style={{ height: '300px', width: '100%', position: 'relative', overflow: 'hidden' }}>
            <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <Box position={[-1.2, 0, 0]} delay={0} />
                <Box position={[0, 0.5, 0.5]} delay={1.5} />
                <Box position={[1.2, -0.2, -0.5]} delay={3} />
                <Box position={[0.5, -0.8, 0.2]} delay={4.5} />
            </Canvas>
            <div style={{
                position: 'absolute',
                inset: 0,
                background: 'radial-gradient(circle at center, transparent 0%, var(--bg) 95%)',
                pointerEvents: 'none'
            }} />
        </div>
    );
}
