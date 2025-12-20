"use client";

import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

function AnimatedSphere() {
    const meshRef = useRef<THREE.Mesh>(null);

    // Create a custom wireframe/pixelated material feel
    const material = useMemo(() => new THREE.MeshStandardMaterial({
        color: '#ffffff',
        wireframe: true,
        transparent: true,
        opacity: 0.1,
    }), []);

    useFrame((state) => {
        if (meshRef.current) {
            meshRef.current.rotation.x = state.clock.getElapsedTime() * 0.2;
            meshRef.current.rotation.y = state.clock.getElapsedTime() * 0.3;
        }
    });

    return (
        <group>
            <mesh ref={meshRef}>
                <sphereGeometry args={[1.5, 32, 32]} />
                <primitive object={material} attach="material" />
            </mesh>

            {/* Secondary inner sphere for depth */}
            <mesh rotation={[Math.PI / 4, 0, 0]}>
                <sphereGeometry args={[1.2, 16, 16]} />
                <meshStandardMaterial color="#ffffff" wireframe={true} transparent={true} opacity={0.05} />
            </mesh>
        </group>
    );
}

export function HeroSphere() {
    return (
        <div style={{ height: '300px', width: '100%', position: 'relative', overflow: 'hidden' }}>
            <Canvas camera={{ position: [0, 0, 5], fov: 45 }}>
                <ambientLight intensity={0.5} />
                <pointLight position={[10, 10, 10]} intensity={1} />
                <AnimatedSphere />
            </Canvas>
            <div style={{
                position: 'absolute',
                inset: 0,
                background: 'radial-gradient(circle at center, transparent 0%, var(--bg) 90%)',
                pointerEvents: 'none'
            }} />
        </div>
    );
}
