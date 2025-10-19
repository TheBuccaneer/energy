import React, { useState, useMemo } from 'react';
import { ArrowRight, Clock, Zap, DollarSign, Leaf } from 'lucide-react';

const DecisionMatrixViz = () => {
  const [selectedZone, setSelectedZone] = useState('DE');
  const [selectedWorkload, setSelectedWorkload] = useState('GEMM');
  const [selectedSize, setSelectedSize] = useState(2048);

  // Zone characteristics
  const zones = {
    DE: {
      name: 'Germany',
      cheapWindow: [1, 6],
      greenWindow: [8, 12],
      avgPrice: 0.09,
      avgCI: 280,
      color: 'bg-yellow-100 border-yellow-500'
    },
    FR: {
      name: 'France',
      cheapWindow: [13, 15],
      greenWindow: [0, 23],
      avgPrice: 0.05,
      avgCI: 25,
      color: 'bg-blue-100 border-blue-500'
    },
    PL: {
      name: 'Poland',
      cheapWindow: [0, 4],
      greenWindow: [0, 4],
      avgPrice: 0.11,
      avgCI: 650,
      color: 'bg-red-100 border-red-500'
    }
  };

  // Device rules
  const getDeviceRule = (workload, size) => {
    if (workload === 'GEMM' && size >= 2048) {
      return {
        device: 'GPU (RTX 5050)',
        rationale: 'Size ≥2048: GPU break-even achieved',
        savings: '37% energy vs CPU',
        color: 'text-green-600'
      };
    }
    return {
      device: 'CPU',
      rationale: workload === 'GEMM' 
        ? 'Size <2048: CPU dominates' 
        : 'GPU never competitive for this workload',
      savings: 'Best energy efficiency',
      color: 'text-blue-600'
    };
  };

  // Timing decision
  const getTimingDecision = (hour, zone) => {
    const zoneInfo = zones[zone];
    const [cheapStart, cheapEnd] = zoneInfo.cheapWindow;
    const [greenStart, greenEnd] = zoneInfo.greenWindow;
    
    const inCheap = hour >= cheapStart && hour <= cheapEnd;
    const inGreen = hour >= greenStart && hour <= greenEnd;

    if (inCheap && inGreen) {
      return {
        action: 'Execute Now',
        icon: Zap,
        color: 'bg-green-500',
        delay: 0,
        reason: 'Optimal window (cheap + green)',
        savings: { cost: 0, co2: 0 }
      };
    }

    if (inCheap) {
      return {
        action: 'Execute Now',
        icon: DollarSign,
        color: 'bg-blue-500',
        delay: 0,
        reason: 'Cheapest window',
        savings: { cost: 0, co2: 0 }
      };
    }

    if (inGreen) {
      return {
        action: 'Execute Now',
        icon: Leaf,
        color: 'bg-green-600',
        delay: 0,
        reason: 'Greenest window',
        savings: { cost: 0, co2: 0 }
      };
    }

    // Calculate delays
    const delayCheap = hour < cheapStart ? cheapStart - hour : (24 - hour) + cheapStart;
    const delayGreen = hour < greenStart ? greenStart - hour : (24 - hour) + greenStart;
    const minDelay = Math.min(delayCheap, delayGreen);

    if (minDelay <= 8) {
      const waitForCheap = delayCheap <= delayGreen;
      const targetHour = (hour + (waitForCheap ? delayCheap : delayGreen)) % 24;
      
      return {
        action: `Wait ${waitForCheap ? delayCheap : delayGreen}h → Execute @ ${String(targetHour).padStart(2, '0')}:00`,
        icon: Clock,
        color: 'bg-orange-500',
        delay: waitForCheap ? delayCheap : delayGreen,
        reason: waitForCheap ? 'Wait for cheap window' : 'Wait for green window',
        savings: {
          cost: minDelay <= 4 ? 20 : 35,
          co2: minDelay <= 4 ? 20 : 35
        }
      };
    }

    return {
      action: 'Execute Now',
      icon: Zap,
      color: 'bg-gray-500',
      delay: 0,
      reason: 'Optimal window too far (>8h)',
      savings: { cost: 0, co2: 0 }
    };
  };

  // Generate hourly decisions
  const hourlyDecisions = useMemo(() => {
    return Array.from({ length: 24 }, (_, hour) => {
      const device = getDeviceRule(selectedWorkload, selectedSize);
      const timing = getTimingDecision(hour, selectedZone);
      
      return {
        hour,
        device: device.device,
        timing,
        totalSavings: {
          cost: timing.savings.cost + (device.device.includes('GPU') ? 15 : 0),
          co2: timing.savings.co2 + (device.device.includes('GPU') ? 20 : 0)
        }
      };
    });
  }, [selectedZone, selectedWorkload, selectedSize]);

  const deviceRule = getDeviceRule(selectedWorkload, selectedSize);
  const currentZone = zones[selectedZone];

  return (
    <div className="p-6 max-w-7xl mx-auto bg-gray-50">
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">
          Carbon & Cost-Aware Decision Matrix
        </h1>
        <p className="text-gray-600">
          Interactive decision support for HPC workload scheduling
        </p>
      </div>

      {/* Controls */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <div className="grid grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Zone
            </label>
            <select
              value={selectedZone}
              onChange={(e) => setSelectedZone(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded"
            >
              {Object.entries(zones).map(([code, info]) => (
                <option key={code} value={code}>
                  {info.name} ({code})
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Workload
            </label>
            <select
              value={selectedWorkload}
              onChange={(e) => setSelectedWorkload(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded"
            >
              <option value="GEMM">GEMM (compute-bound)</option>
              <option value="SPMV">SpMV (irregular)</option>
              <option value="REDUCTION">Reduction (compute)</option>
              <option value="STREAM">STREAM (memory-bound)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Matrix Size (N)
            </label>
            <select
              value={selectedSize}
              onChange={(e) => setSelectedSize(Number(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded"
            >
              {[64, 128, 256, 512, 1024, 2048, 4096].map(size => (
                <option key={size} value={size}>{size}</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Zone Info Card */}
      <div className={`${currentZone.color} border-2 p-4 rounded-lg mb-6`}>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h3 className="font-semibold text-gray-800 mb-2">Zone Characteristics</h3>
            <div className="text-sm space-y-1">
              <div className="flex items-center gap-2">
                <DollarSign size={16} className="text-blue-600" />
                <span>Avg Price: €{currentZone.avgPrice}/kWh</span>
              </div>
              <div className="flex items-center gap-2">
                <Leaf size={16} className="text-green-600" />
                <span>Avg CI: {currentZone.avgCI} gCO₂/kWh</span>
              </div>
            </div>
          </div>
          <div>
            <h3 className="font-semibold text-gray-800 mb-2">Optimal Windows</h3>
            <div className="text-sm space-y-1">
              <div>💰 Cheapest: {String(currentZone.cheapWindow[0]).padStart(2, '0')}:00-{String(currentZone.cheapWindow[1]).padStart(2, '0')}:00</div>
              <div>🌱 Greenest: {String(currentZone.greenWindow[0]).padStart(2, '0')}:00-{String(currentZone.greenWindow[1]).padStart(2, '0')}:00</div>
            </div>
          </div>
        </div>
      </div>

      {/* Device Decision Card */}
      <div className="bg-white p-4 rounded-lg shadow mb-6">
        <h3 className="font-semibold text-gray-800 mb-3">Device Selection Rule</h3>
        <div className="flex items-center gap-4">
          <div className={`text-2xl font-bold ${deviceRule.color}`}>
            {deviceRule.device}
          </div>
          <ArrowRight size={24} className="text-gray-400" />
          <div className="flex-1">
            <div className="text-sm text-gray-700">{deviceRule.rationale}</div>
            <div className="text-xs text-gray-500 mt-1">{deviceRule.savings}</div>
          </div>
        </div>
      </div>

      {/* Hourly Timeline */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="font-semibold text-gray-800 mb-4">24-Hour Decision Timeline</h3>
        
        <div className="space-y-2">
          {hourlyDecisions.map(({ hour, timing, totalSavings }) => {
            const Icon = timing.icon;
            
            return (
              <div
                key={hour}
                className="flex items-center gap-3 p-3 rounded border border-gray-200 hover:bg-gray-50 transition"
              >
                <div className="w-16 text-center font-mono font-semibold">
                  {String(hour).padStart(2, '0')}:00
                </div>
                
                <div className={`${timing.color} text-white p-2 rounded`}>
                  <Icon size={20} />
                </div>
                
                <div className="flex-1">
                  <div className="font-medium text-gray-800">{timing.action}</div>
                  <div className="text-xs text-gray-500">{timing.reason}</div>
                </div>
                
                {timing.delay > 0 && (
                  <div className="text-right">
                    <div className="text-sm font-semibold text-orange-600">
                      {timing.delay}h delay
                    </div>
                    <div className="text-xs text-gray-500">
                      💰 {totalSavings.cost}% cost | 🌱 {totalSavings.co2}% CO₂
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Summary Statistics */}
      <div className="mt-6 bg-white p-4 rounded-lg shadow">
        <h3 className="font-semibold text-gray-800 mb-3">Summary Statistics</h3>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold text-blue-600">
              {hourlyDecisions.filter(d => d.timing.delay === 0).length}
            </div>
            <div className="text-xs text-gray-600">Execute Immediately</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-orange-600">
              {hourlyDecisions.filter(d => d.timing.delay > 0).length}
            </div>
            <div className="text-xs text-gray-600">Wait for Optimal Window</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-600">
              {Math.round(hourlyDecisions.reduce((sum, d) => sum + d.totalSavings.co2, 0) / 24)}%
            </div>
            <div className="text-xs text-gray-600">Avg CO₂ Savings Potential</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DecisionMatrixViz;