<template>
  <!-- Multi-GPU Monitor Component (Custom Extension) -->
  <!-- This is a standalone component for multi-GPU display -->
  <!-- Can be imported into Monitor.vue with minimal changes -->
  <div class="multi-gpu-monitor" v-if="hasMultipleGpus">
    <div class="gpu-header">
      <span class="gpu-count">{{ gpuCount }} GPUs 并行</span>
      <span class="gpu-aggregate">
        总显存: {{ totalMemoryUsed.toFixed(1) }} / {{ totalMemoryTotal.toFixed(1) }} GB
      </span>
    </div>
    
    <div class="gpu-cards">
      <div 
        v-for="gpu in gpus" 
        :key="gpu.index" 
        class="gpu-card"
      >
        <div class="gpu-card-header">
          <span class="gpu-index">GPU {{ gpu.index }}</span>
          <span class="gpu-name">{{ gpu.name }}</span>
        </div>
        
        <div class="gpu-metric">
          <span class="metric-label">显存</span>
          <el-progress 
            :percentage="gpu.memoryPercent" 
            :stroke-width="12"
            :format="() => `${gpu.memoryUsed}/${gpu.memoryTotal} GB`"
          />
        </div>
        
        <div class="gpu-metric">
          <span class="metric-label">利用率</span>
          <el-progress 
            :percentage="gpu.utilization" 
            :stroke-width="12"
            :status="gpu.utilization > 90 ? 'success' : ''"
          />
        </div>
        
        <div class="gpu-temp">
          <span class="temp-value" :class="getTempClass(gpu.temperature)">
            {{ gpu.temperature }}°C
          </span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useSystemStore } from '@/stores/system'

const systemStore = useSystemStore()

// Check if we have multiple GPUs
const hasMultipleGpus = computed(() => {
  const info = systemStore.gpuInfo as any
  return info.numGpus && info.numGpus > 1
})

const gpuCount = computed(() => {
  const info = systemStore.gpuInfo as any
  return info.numGpus || 1
})

const gpus = computed(() => {
  const info = systemStore.gpuInfo as any
  return info.gpus || []
})

const totalMemoryUsed = computed(() => {
  const info = systemStore.gpuInfo as any
  return info.memoryUsed || 0
})

const totalMemoryTotal = computed(() => {
  const info = systemStore.gpuInfo as any
  return info.memoryTotal || 0
})

function getTempClass(temp: number): string {
  if (temp < 60) return 'cool'
  if (temp < 80) return 'warm'
  return 'hot'
}
</script>

<style scoped>
.multi-gpu-monitor {
  margin-bottom: 1.5rem;
}

.gpu-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 1rem;
  padding: 0.5rem 1rem;
  background: linear-gradient(135deg, rgba(64, 158, 255, 0.1), rgba(103, 194, 58, 0.1));
  border-radius: 8px;
}

.gpu-count {
  font-weight: 600;
  color: var(--el-color-primary);
  font-size: 1.1rem;
}

.gpu-aggregate {
  color: var(--el-text-color-secondary);
  font-size: 0.9rem;
}

.gpu-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
}

.gpu-card {
  background: var(--el-bg-color);
  border: 1px solid var(--el-border-color-lighter);
  border-radius: 12px;
  padding: 1rem;
  transition: all 0.3s ease;
}

.gpu-card:hover {
  border-color: var(--el-color-primary-light-5);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.gpu-card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--el-border-color-lighter);
}

.gpu-index {
  font-weight: 600;
  color: var(--el-color-primary);
  background: var(--el-color-primary-light-9);
  padding: 0.2rem 0.6rem;
  border-radius: 4px;
  font-size: 0.85rem;
}

.gpu-name {
  font-size: 0.8rem;
  color: var(--el-text-color-secondary);
  max-width: 150px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.gpu-metric {
  margin-bottom: 0.75rem;
}

.metric-label {
  display: block;
  font-size: 0.8rem;
  color: var(--el-text-color-secondary);
  margin-bottom: 0.25rem;
}

.gpu-temp {
  text-align: right;
  margin-top: 0.5rem;
}

.temp-value {
  font-weight: 600;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
}

.temp-value.cool {
  color: #67c23a;
  background: rgba(103, 194, 58, 0.1);
}

.temp-value.warm {
  color: #e6a23c;
  background: rgba(230, 162, 60, 0.1);
}

.temp-value.hot {
  color: #f56c6c;
  background: rgba(245, 108, 108, 0.1);
}
</style>

