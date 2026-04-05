// BrainRenderer — 3D neural network with soft lighting
import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, extend } from '@react-three/fiber';
import { OrbitControls, Effects } from '@react-three/drei';
import * as THREE from 'three';
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { useStore } from '../store';

extend({ UnrealBloomPass });

const DEPTH_COLORS = [
  new THREE.Color('#22b8cf'), // input (teal)
  new THREE.Color('#5b6cf0'), // processing (soft blue)
  new THREE.Color('#8b5cf6'), // abstract (lavender)
  new THREE.Color('#f06292'), // meta (soft pink)
];

function Nodes() {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const topology = useStore((s) => s.topology);
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const { count, colorArray, sizeArray } = useMemo(() => {
    if (!topology) return { count: 0, colorArray: new Float32Array(0), sizeArray: new Float32Array(0) };
    const n = topology.nodes.length;
    const colors = new Float32Array(n * 3);
    const sizes = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const node = topology.nodes[i];
      const baseColor = DEPTH_COLORS[Math.min(node.depth, 3)] || DEPTH_COLORS[0];
      const energy = node.active ? 1.6 : 0.4;
      colors[i * 3] = baseColor.r * energy;
      colors[i * 3 + 1] = baseColor.g * energy;
      colors[i * 3 + 2] = baseColor.b * energy;
      sizes[i] = node.active ? 0.22 : 0.09;
    }
    return { count: n, colorArray: colors, sizeArray: sizes };
  }, [topology]);

  useEffect(() => {
    if (!meshRef.current || !topology) return;
    const mesh = meshRef.current;

    for (let i = 0; i < topology.nodes.length; i++) {
      const node = topology.nodes[i];
      dummy.position.set(node.pos[0], node.pos[1], node.pos[2]);
      const scale = sizeArray[i];
      dummy.scale.set(scale, scale, scale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
      mesh.setColorAt(i, new THREE.Color(colorArray[i * 3], colorArray[i * 3 + 1], colorArray[i * 3 + 2]));
    }
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [topology, colorArray, sizeArray, dummy]);

  useFrame(({ clock }) => {
    if (!meshRef.current || !topology) return;
    const t = clock.getElapsedTime();
    const mesh = meshRef.current;

    for (let i = 0; i < Math.min(topology.nodes.length, 500); i++) {
      const node = topology.nodes[i];
      if (!node.active) continue;
      const pulse = 1.0 + 0.15 * Math.sin(t * 2 + node.oscillation_phase * Math.PI * 2);
      const base = sizeArray[i] || 0.22;
      dummy.position.set(node.pos[0], node.pos[1], node.pos[2]);
      dummy.scale.set(base * pulse, base * pulse, base * pulse);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }
    mesh.instanceMatrix.needsUpdate = true;
  });

  if (count === 0) return null;

  return (
    <instancedMesh ref={meshRef} args={[undefined, undefined, count]}>
      <sphereGeometry args={[1, 16, 12]} />
      <meshStandardMaterial
        toneMapped={false}
        emissive="#5b6cf0"
        emissiveIntensity={0.6}
        roughness={0.5}
        metalness={0.05}
      />
    </instancedMesh>
  );
}

function Edges() {
  const linesRef = useRef<THREE.LineSegments>(null);
  const topology = useStore((s) => s.topology);

  const geometry = useMemo(() => {
    if (!topology || topology.links.length === 0) return null;

    const nodeMap = new Map<number, [number, number, number]>();
    for (const node of topology.nodes) {
      nodeMap.set(node.id, node.pos);
    }

    const positions: number[] = [];
    const colors: number[] = [];

    for (const link of topology.links) {
      const src = nodeMap.get(link.s);
      const tgt = nodeMap.get(link.t);
      if (!src || !tgt) continue;

      positions.push(src[0], src[1], src[2]);
      positions.push(tgt[0], tgt[1], tgt[2]);

      const intensity = Math.min(Math.abs(link.w) * 3, 0.6);
      if (link.w > 0) {
        colors.push(0.35, 0.42, 0.94);
        colors.push(0.35 * intensity, 0.42 * intensity, 0.94 * intensity);
      } else {
        colors.push(0.94, 0.38, 0.57);
        colors.push(0.94 * intensity, 0.38 * intensity, 0.57 * intensity);
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    return geo;
  }, [topology]);

  if (!geometry) return null;

  return (
    <lineSegments ref={linesRef} geometry={geometry}>
      <lineBasicMaterial vertexColors transparent opacity={0.12} />
    </lineSegments>
  );
}

function BrainScene() {
  return (
    <>
      <color attach="background" args={['#f0f2f8']} />
      <fog attach="fog" args={['#f0f2f8', 25, 65]} />
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={0.8} color="#5b6cf0" distance={40} />
      <pointLight position={[-10, -5, -10]} intensity={0.4} color="#8b5cf6" distance={30} />
      <pointLight position={[0, -8, 5]} intensity={0.3} color="#22b8cf" distance={25} />
      <Nodes />
      <Edges />
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        minDistance={3}
        maxDistance={50}
        autoRotate
        autoRotateSpeed={0.3}
      />
      <Effects>
        {/* @ts-ignore */}
        <unrealBloomPass args={[new THREE.Vector2(window.innerWidth, window.innerHeight), 0.9, 0.1, 0.6]} />
      </Effects>
    </>
  );
}

export default function BrainRenderer() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 6, 18], fov: 55, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: false, toneMapping: THREE.ACESFilmicToneMapping, toneMappingExposure: 1.0 }}
      >
        <BrainScene />
      </Canvas>
    </div>
  );
}
