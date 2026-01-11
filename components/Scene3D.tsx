import React, { useRef, useMemo, useEffect } from 'react';
import { useFrame } from '@react-three/fiber';
import { Sphere, Line, Text, OrbitControls, Grid, Html } from '@react-three/drei';
import * as THREE from 'three';
import { Neuron, Connection } from '../types';

interface Scene3DProps {
  neurons: Neuron[];
  connections: Connection[];
  selectedId: string | null;
  onSelect: (id: string, type: 'neuron' | 'connection') => void;
  currentInput: number[] | undefined;
  batchProgress: number;
  batchSize: number;
}

// Particle system to visualize data flow "Animation"
const DataParticles = ({ connections }: { connections: Connection[] }) => {
    const particleCount = 60; // Enough to look busy but clean
    const meshRef = useRef<THREE.InstancedMesh>(null);
    const dummy = useMemo(() => new THREE.Object3D(), []);

    const particles = useMemo(() => {
        return new Array(particleCount).fill(0).map(() => ({
            offset: Math.random(),
            speed: 0.5 + Math.random() * 0.5,
            connectionIndex: Math.floor(Math.random() * connections.length)
        }));
    }, [connections.length]);

    useFrame((state, delta) => {
        if (!meshRef.current || connections.length === 0) return;

        particles.forEach((particle, i) => {
            const conn = connections[particle.connectionIndex];
            if(!conn) return;

            // Animate offset
            particle.offset += delta * particle.speed;
            if (particle.offset > 1) {
                particle.offset = 0;
                particle.connectionIndex = Math.floor(Math.random() * connections.length);
            }

            const start = new THREE.Vector3(...conn.sourcePos);
            const end = new THREE.Vector3(...conn.targetPos);
            const pos = start.lerp(end, particle.offset);
            
            dummy.position.copy(pos);
            const scale = 0.08;
            dummy.scale.setScalar(scale);
            dummy.updateMatrix();
            meshRef.current!.setMatrixAt(i, dummy.matrix);
        });
        meshRef.current.instanceMatrix.needsUpdate = true;
    });

    return (
        <instancedMesh ref={meshRef} args={[undefined, undefined, particleCount]}>
            <sphereGeometry args={[1, 8, 8]} />
            <meshBasicMaterial color="#fbbf24" toneMapped={false} />
        </instancedMesh>
    );
};

// 3D Board showing the raw input vector and batch info
const InputDataBoard = ({ inputs, batchProgress, batchSize, inputLayerX }: { inputs: number[] | undefined, batchProgress: number, batchSize: number, inputLayerX: number }) => {
    if (!inputs) return null;

    // Position board to the left of the input layer
    const boardX = inputLayerX - 5;
    
    // Calculate local target x for the lines (where the input neurons are relative to the board)
    // Input neurons are at world x = inputLayerX
    // Board is at world x = boardX
    // Local X of input neurons = inputLayerX - boardX = 5
    // We want lines to stop slightly before the neurons (radius 0.5) => 4.5
    const lineTargetX = 4.5;

    return (
        <group position={[boardX, 0, 0]}>
            {/* The Board Backing */}
            <mesh position={[0, 0, -0.1]}>
                <boxGeometry args={[3, 3.5, 0.2]} />
                <meshStandardMaterial color="#1e293b" metalness={0.8} roughness={0.2} />
            </mesh>
            
            {/* Header */}
            <Text position={[0, 1.4, 0.11]} fontSize={0.2} color="#94a3b8">
                CURRENT INPUT
            </Text>
            
            {/* The Matrix Vector Visualization */}
            <Html transform position={[0, 0.2, 0.11]} scale={0.5} style={{ pointerEvents: 'none' }}>
                <div className="flex items-center gap-2 font-mono text-3xl text-white">
                    <div className="text-slate-500">x =</div>
                    <div className="flex flex-col border-l-4 border-r-4 border-slate-400 px-4 py-2 gap-2 rounded-sm bg-slate-900/50">
                        {inputs.map((val, i) => (
                             <div key={i} className="text-emerald-400 font-bold">
                                 {val.toFixed(2)}
                             </div>
                        ))}
                    </div>
                </div>
            </Html>
            
            {/* Batch Status */}
            <group position={[0, -1, 0.11]}>
                <Text fontSize={0.15} color="#64748b" position={[0, 0.2, 0]}>BATCH PROGRESS</Text>
                {/* Progress Bar Background */}
                <mesh position={[0, -0.1, 0]}>
                    <planeGeometry args={[2.5, 0.2]} />
                    <meshBasicMaterial color="#334155" />
                </mesh>
                {/* Progress Bar Fill */}
                <mesh position={[ -1.25 + (batchProgress / batchSize) * 2.5 * 0.5, -0.1, 0.01]}>
                    <planeGeometry args={[(batchProgress / batchSize) * 2.5, 0.2]} />
                    <meshBasicMaterial color={batchProgress === 0 ? "#10b981" : "#3b82f6"} />
                </mesh>
                <Text fontSize={0.15} color="#e2e8f0" position={[0, -0.4, 0]}>
                    {batchProgress} / {batchSize}
                </Text>
            </group>

            {/* Connectors to Input Layer (Visual only) */}
            <Line points={[[1.5, 0.5, 0], [lineTargetX, 1, 0]]} color="#475569" transparent opacity={0.3} lineWidth={1} />
            <Line points={[[1.5, -0.5, 0], [lineTargetX, -1, 0]]} color="#475569" transparent opacity={0.3} lineWidth={1} />
        </group>
    );
};

interface NeuronMeshProps {
  data: Neuron;
  isSelected: boolean;
  onClick: () => void;
}

const NeuronMesh: React.FC<NeuronMeshProps> = ({ data, isSelected, onClick }) => {
  const meshRef = useRef<THREE.Mesh>(null);
  
  const color = useMemo(() => {
    const c = new THREE.Color();
    // Activation mapping: Dark Blue (0) -> Bright Teal (1)
    c.setHSL(0.55, 0.8, 0.1 + data.activation * 0.8);
    return c;
  }, [data.activation]);

  useFrame((state) => {
    if (meshRef.current) {
        if (isSelected) {
            meshRef.current.scale.setScalar(1.2 + 0.1 * Math.sin(state.clock.elapsedTime * 10));
        } else {
            meshRef.current.scale.setScalar(1);
        }
    }
  });

  return (
    <group position={data.position}>
      <Sphere ref={meshRef} args={[0.5, 32, 32]} onClick={(e) => { e.stopPropagation(); onClick(); }}>
        <meshStandardMaterial 
          color={isSelected ? '#fde047' : color} 
          emissive={isSelected ? '#fde047' : color}
          emissiveIntensity={0.3}
          roughness={0.4}
          metalness={0.7}
        />
      </Sphere>
      
      {/* Neuron Label */}
      <Text position={[0, -0.8, 0]} fontSize={0.25} color="#64748b" anchorX="center" anchorY="top">
        {data.layerIndex === 0 ? `In ${data.neuronIndex + 1}` : `n${data.layerIndex}.${data.neuronIndex}`}
      </Text>

      {/* MATRIX DISPLAY ABOVE NODE */}
      <Html 
        position={[0, 0.8, 0]} 
        center 
        transform
        distanceFactor={8}
        className="pointer-events-none select-none"
        style={{ transition: 'all 0.1s', opacity: 1 }}
        zIndexRange={[100, 0]}
      >
          <div className="flex flex-col items-center">
             <div className="flex gap-1 bg-slate-900/90 backdrop-blur-md border border-slate-600 px-3 py-2 rounded shadow-2xl">
                 
                 {/* Z Value (Pre-activation) - Only for hidden/output */}
                 {data.layerIndex > 0 && (
                     <div className="flex flex-col items-center border-r border-slate-700 pr-2 mr-2">
                        <span className="text-[10px] text-slate-400 mb-0.5">z</span>
                        <div className="font-mono text-sm text-slate-300 border-l-2 border-r-2 border-slate-500 px-1">
                            {data.preActivation.toFixed(2)}
                        </div>
                     </div>
                 )}

                 {/* A Value (Activation) */}
                 <div className="flex flex-col items-center">
                    <span className="text-[10px] text-indigo-300 mb-0.5">a</span>
                    <div className="font-mono text-sm text-white font-bold border-l-2 border-r-2 border-white px-1">
                        {data.activation.toFixed(2)}
                    </div>
                 </div>

             </div>
             
             {/* Gradient Pill below if training */}
             {Math.abs(data.gradient) > 0.001 && (
                 <div className="mt-1 bg-rose-500/20 text-rose-300 text-[10px] px-1.5 py-0.5 rounded-full border border-rose-500/50">
                     δ {data.gradient.toFixed(3)}
                 </div>
             )}
          </div>
      </Html>

    </group>
  );
};

interface ConnectionLineProps {
  data: Connection;
  isSelected: boolean;
  onClick: () => void;
}

const ConnectionLine: React.FC<ConnectionLineProps> = ({ data, isSelected, onClick }) => {
  const lineWidth = isSelected ? 3 : Math.max(0.5, Math.abs(data.weight) * 2);
  const color = data.weight > 0 ? '#4ade80' : '#f87171'; // Green (+) / Red (-)
  
  return (
    <Line
      points={[data.sourcePos, data.targetPos]}
      color={isSelected ? '#fde047' : color}
      lineWidth={lineWidth}
      transparent
      opacity={isSelected ? 1 : 0.3} // Lower opacity to make particles visible
      onClick={(e) => { e.stopPropagation(); onClick(); }}
    />
  );
};

const Scene3D: React.FC<Scene3DProps> = ({ neurons, connections, selectedId, onSelect, currentInput, batchProgress, batchSize }) => {
  // Determine X position of input layer to attach board correctly
  const inputLayerX = useMemo(() => {
    // Input layer is index 0. Find the first neuron or default.
    const firstNeuron = neurons.find(n => n.layerIndex === 0);
    return firstNeuron ? firstNeuron.position[0] : -6;
  }, [neurons]);

  return (
    <>
      <color attach="background" args={['#0f172a']} /> {/* Slate 950 match body */}
      
      <ambientLight intensity={0.7} />
      <directionalLight position={[10, 10, 5]} intensity={1} />
      <directionalLight position={[-10, 20, 10]} intensity={0.5} />
      
      <Grid 
        infiniteGrid 
        fadeDistance={40} 
        sectionColor="#334155" 
        cellColor="#1e293b" 
        position={[0, -6, 0]} 
      />

      <group>
        <InputDataBoard 
            inputs={currentInput} 
            batchProgress={batchProgress} 
            batchSize={batchSize} 
            inputLayerX={inputLayerX}
        />

        {connections.map((conn) => (
          <ConnectionLine 
            key={conn.id} 
            data={conn} 
            isSelected={selectedId === conn.id}
            onClick={() => onSelect(conn.id, 'connection')} 
          />
        ))}
        
        {neurons.map((neuron) => (
          <NeuronMesh 
            key={neuron.id} 
            data={neuron} 
            isSelected={selectedId === neuron.id}
            onClick={() => onSelect(neuron.id, 'neuron')}
          />
        ))}

        {/* Animation Particles */}
        <DataParticles connections={connections} />
      </group>

      <OrbitControls makeDefault enableDamping dampingFactor={0.1} />
    </>
  );
};

export default Scene3D;