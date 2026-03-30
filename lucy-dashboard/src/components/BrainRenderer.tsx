// BrainRenderer — 3D neural network visualization using Three.js instanced meshes
import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { useStore } from '../store';

// Color palette for depth layers
const DEPTH_COLORS = [
  new THREE.Color('#00ddff'), // 0 = input (cyan)
  new THREE.Color('#4466ff'), // 1 = processing (blue)
  new THREE.Color('#aa44ff'), // 2 = abstract (purple)
  new THREE.Color('#ff44aa'), // 3 = meta (pink)
];

const ACTIVE_BOOST = 1.8;
const INACTIVE_ALPHA = 0.3;

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
      const brightness = node.active ? ACTIVE_BOOST : INACTIVE_ALPHA;
      colors[i * 3] = baseColor.r * brightness;
      colors[i * 3 + 1] = baseColor.g * brightness;
      colors[i * 3 + 2] = baseColor.b * brightness;
      sizes[i] = node.active ? 0.12 : 0.05;
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

  // Gentle pulsing based on oscillation phase
  useFrame(({ clock }) => {
    if (!meshRef.current || !topology) return;
    const t = clock.getElapsedTime();
    const mesh = meshRef.current;

    for (let i = 0; i < Math.min(topology.nodes.length, 200); i++) {
      const node = topology.nodes[i];
      if (!node.active) continue;
      const pulse = 1.0 + 0.15 * Math.sin(t * 2 + node.oscillation_phase * Math.PI * 2);
      const base = sizeArray[i] || 0.12;
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
      <sphereGeometry args={[1, 8, 6]} />
      <meshStandardMaterial
        toneMapped={false}
        emissive="#4466ff"
        emissiveIntensity={0.5}
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

      // Color: green for positive weights, red for negative
      const intensity = Math.min(Math.abs(link.w) * 3, 1.0);
      if (link.w > 0) {
        colors.push(0, intensity * 0.8, intensity * 0.5);
        colors.push(0, intensity * 0.8, intensity * 0.5);
      } else {
        colors.push(intensity * 0.8, 0, intensity * 0.3);
        colors.push(intensity * 0.8, 0, intensity * 0.3);
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
      <lineBasicMaterial vertexColors transparent opacity={0.15} />
    </lineSegments>
  );
}

function BrainScene() {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={0.8} color="#4466ff" />
      <pointLight position={[-10, -5, -10]} intensity={0.4} color="#aa44ff" />
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
    </>
  );
}

export default function BrainRenderer() {
  return (
    <div className="w-full h-full">
      <Canvas
        camera={{ position: [0, 5, 15], fov: 60, near: 0.1, far: 200 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <BrainScene />
      </Canvas>
    </div>
  );
}
