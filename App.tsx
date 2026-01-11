import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import Scene3D from './components/Scene3D';
import Controls from './components/Controls';
import InfoPanel from './components/InfoPanel';
import { MiniNeuralNet, generateData } from './services/neuralNet';
import { NetworkConfig, ActivationType, DatasetType, Neuron, Connection, TrainingStats } from './types';

const INITIAL_CONFIG: NetworkConfig = {
  learningRate: 0.1,
  dataset: DatasetType.XOR,
  batchSize: 1,
  layers: [
    { neurons: 2, activation: ActivationType.RELU },
    { neurons: 3, activation: ActivationType.TANH },
    { neurons: 1, activation: ActivationType.SIGMOID }
  ]
};

const App: React.FC = () => {
  const [config, setConfig] = useState<NetworkConfig>(INITIAL_CONFIG);
  
  // Training State
  const [isPlaying, setIsPlaying] = useState(false);
  const [speedMode, setSpeedMode] = useState<'step' | 'fast'>('fast');
  
  const [trainingStats, setTrainingStats] = useState<TrainingStats>({ 
    epoch: 0, 
    loss: 0, 
    accuracy: 0,
    currentInputIndex: 0,
    history: [],
    batchProgress: 0,
    batchSize: 1
  });
  
  const [selectedElement, setSelectedElement] = useState<{id: string, type: 'neuron'|'connection'} | null>(null);

  // Data & Model Refs
  const nnRef = useRef<MiniNeuralNet>(new MiniNeuralNet(INITIAL_CONFIG.layers, INITIAL_CONFIG.learningRate, INITIAL_CONFIG.batchSize));
  const dataRef = useRef(generateData(INITIAL_CONFIG.dataset));
  
  // Visualization State
  const [visData, setVisData] = useState<{ neurons: Neuron[], connections: Connection[] }>({ neurons: [], connections: [] });
  const animationFrameRef = useRef<number>();
  const stepIndexRef = useRef(0);

  const resetNetwork = useCallback(() => {
    setIsPlaying(false);
    nnRef.current = new MiniNeuralNet(config.layers, config.learningRate, config.batchSize);
    dataRef.current = generateData(config.dataset);
    stepIndexRef.current = 0;
    
    setTrainingStats({ 
        epoch: 0, 
        loss: 0, 
        accuracy: 0, 
        currentInputIndex: 0, 
        history: [],
        batchProgress: 0,
        batchSize: config.batchSize
    });
    
    setVisData(nnRef.current.getVisualizationData());
  }, [config]);

  // Initial Load
  useEffect(() => {
    resetNetwork();
  }, [resetNetwork]);

  // Sync Params
  useEffect(() => {
    if(nnRef.current) {
        nnRef.current.updateConfig(config.learningRate, config.batchSize);
    }
  }, [config.learningRate, config.batchSize]);

  // The Logic Engine
  const performStep = useCallback(() => {
    const nn = nnRef.current;
    const data = dataRef.current;
    const index = stepIndexRef.current % data.length;
    
    // Single Step Train
    const row = data[index];
    nn.train(row.inputs, row.target);
    
    // Update Stats
    stepIndexRef.current++;
    const epoch = Math.floor(stepIndexRef.current / data.length);
    
    // Calculate loss for this specific step
    const output = nn.activations[nn.activations.length-1];
    const loss = Math.pow(row.target[0] - output[0], 2);
    
    // Return data for state updates
    return { epoch, loss, index, nn, batchCount: nn.currentBatchCount };
  }, []);

  // Visual Update Loop
  const gameLoop = useCallback(() => {
    if (!isPlaying) return;
    
    if (speedMode === 'fast') {
        let latestStats;
        // If batch size is large, we might want to do more steps to see progress
        const stepsPerFrame = config.batchSize > 1 ? config.batchSize : 4; 
        
        for(let i=0; i<stepsPerFrame; i++) { 
            latestStats = performStep();
        }

        if (latestStats) {
             setTrainingStats(prev => {
                const history = [...prev.history];
                // Only update history periodically to save frames
                if (latestStats.index === 0) { // New Epoch started
                    history.push({ epoch: latestStats.epoch, loss: latestStats.loss });
                    if(history.length > 50) history.shift();
                }
                return {
                    ...prev,
                    epoch: latestStats.epoch,
                    loss: latestStats.loss,
                    currentInputIndex: latestStats.index,
                    history,
                    batchProgress: latestStats.batchCount,
                    batchSize: config.batchSize
                };
            });
            setVisData(latestStats.nn.getVisualizationData());
        }
        animationFrameRef.current = requestAnimationFrame(gameLoop);
    } 
  }, [isPlaying, speedMode, performStep, config.batchSize]);

  useEffect(() => {
    if (isPlaying && speedMode === 'fast') {
      gameLoop();
    } else {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    }
    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isPlaying, speedMode, gameLoop]);

  // Handler for Manual "Step" button
  const handleManualStep = () => {
      setIsPlaying(false); // Pause if playing
      const stats = performStep();
      
      setVisData(stats.nn.getVisualizationData());
      setTrainingStats(prev => {
          const history = [...prev.history];
          return {
            ...prev,
            epoch: stats.epoch,
            loss: stats.loss,
            currentInputIndex: stats.index,
            history,
            batchProgress: stats.batchCount,
            batchSize: config.batchSize
        };
      });
  };

  return (
    <div className="w-full h-screen relative bg-slate-950 overflow-hidden">
      
      {/* 3D View */}
      <div className="absolute inset-0 z-0">
        <Canvas camera={{ position: [0, 0, 35], fov: 35 }}>
            <Scene3D 
                neurons={visData.neurons} 
                connections={visData.connections}
                selectedId={selectedElement?.id || null}
                onSelect={(id, type) => setSelectedElement({id, type})}
                currentInput={dataRef.current[trainingStats.currentInputIndex]?.inputs}
                batchProgress={trainingStats.batchProgress}
                batchSize={trainingStats.batchSize}
            />
        </Canvas>
      </div>

      {/* Left Panel: Controls & Data */}
      <Controls 
        config={config} 
        setConfig={setConfig} 
        isPlaying={isPlaying} 
        speedMode={speedMode}
        setSpeedMode={setSpeedMode}
        setIsPlaying={setIsPlaying}
        onStep={handleManualStep}
        onReset={resetNetwork}
        data={dataRef.current}
        currentInputIndex={trainingStats.currentInputIndex}
      />
      
      {/* Right Panel: Math Inspector */}
      <InfoPanel 
        stats={trainingStats}
        config={config}
        selectedElement={selectedElement}
        neurons={visData.neurons}
        connections={visData.connections}
      />
    </div>
  );
};

export default App;