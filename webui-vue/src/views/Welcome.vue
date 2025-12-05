<template>
  <div class="welcome-container">
    <!-- Hero Section -->
    <div class="hero-section">
      <div class="hero-content">
        <div class="logo-area">
          <div class="logo-icon">
            <span class="logo-text">N</span>
          </div>
        </div>
        <h1 class="main-title">
          <span class="title-gradient">None</span> Trainer
        </h1>
        <p class="tagline">Z-Image Turbo LoRA 训练工作室</p>
        <p class="description">
          基于 <strong>AC-RF</strong>（锚点耦合整流流）算法的高效 LoRA 微调工具
        </p>
        
        <div class="feature-tags">
          <span class="tag"><el-icon><Lightning /></el-icon> 10步快速推理</span>
          <span class="tag"><el-icon><Cpu /></el-icon> 自动硬件优化</span>
          <span class="tag"><el-icon><TrendCharts /></el-icon> 实时训练监控</span>
        </div>
      </div>
      
      <div class="hero-decoration">
        <div class="deco-circle c1"></div>
        <div class="deco-circle c2"></div>
        <div class="deco-circle c3"></div>
      </div>
    </div>

    <div class="dashboard-grid">
      <!-- Quick Actions -->
      <div class="card glass-card quick-actions">
        <h3 class="card-title">
          <el-icon><Operation /></el-icon>
          快速开始
        </h3>
        <div class="action-buttons">
          <div class="action-item" @click="$router.push('/dataset')">
            <div class="action-icon dataset">
              <el-icon><Picture /></el-icon>
            </div>
            <div class="action-info">
              <span class="action-name">数据集管理</span>
              <span class="action-desc">导入图片、生成缓存、Ollama 标注</span>
            </div>
            <el-icon class="arrow"><ArrowRight /></el-icon>
          </div>
          <div class="action-item" @click="$router.push('/config')">
            <div class="action-icon config">
              <el-icon><Setting /></el-icon>
            </div>
            <div class="action-info">
              <span class="action-name">训练配置</span>
              <span class="action-desc">AC-RF 参数、LoRA 设置、优化器</span>
            </div>
            <el-icon class="arrow"><ArrowRight /></el-icon>
          </div>
          <div class="action-item" @click="$router.push('/training')">
            <div class="action-icon train">
              <el-icon><VideoPlay /></el-icon>
            </div>
            <div class="action-info">
              <span class="action-name">开始训练</span>
              <span class="action-desc">实时 Loss 曲线、进度监控</span>
            </div>
            <el-icon class="arrow"><ArrowRight /></el-icon>
          </div>
          <div class="action-item" @click="$router.push('/generation')">
            <div class="action-icon generate">
              <el-icon><MagicStick /></el-icon>
            </div>
            <div class="action-info">
              <span class="action-name">图像生成</span>
              <span class="action-desc">测试训练好的 LoRA 模型</span>
            </div>
            <el-icon class="arrow"><ArrowRight /></el-icon>
          </div>
        </div>
      </div>

      <!-- System Status -->
      <div class="card glass-card system-status">
        <h3 class="card-title">
          <el-icon><Monitor /></el-icon>
          系统状态
          <el-tag v-if="wsConnected" type="success" size="small" effect="plain" class="status-tag">在线</el-tag>
          <el-tag v-else type="danger" size="small" effect="plain" class="status-tag">离线</el-tag>
        </h3>
        
        <div class="status-grid" v-if="hasSystemInfo">
          <div class="status-item">
            <span class="label">Python</span>
            <span class="value">{{ systemInfo.python }}</span>
          </div>
          <div class="status-item">
            <span class="label">PyTorch</span>
            <span class="value">{{ systemInfo.pytorch }}</span>
          </div>
          <div class="status-item">
            <span class="label">CUDA</span>
            <span class="value">{{ systemInfo.cuda }}</span>
          </div>
          <div class="status-item">
            <span class="label">Diffusers</span>
            <span class="value">{{ systemInfo.diffusers }}</span>
          </div>
          <div class="status-item">
            <span class="label">Accelerate</span>
            <span class="value">{{ systemInfo.accelerate || 'N/A' }}</span>
          </div>
          <div class="status-item">
            <span class="label">Transformers</span>
            <span class="value">{{ systemInfo.transformers || 'N/A' }}</span>
          </div>
        </div>
        <div v-else class="loading-placeholder">
          <el-icon class="is-loading"><Loading /></el-icon>
          <span>连接中...</span>
        </div>
      </div>

      <!-- Model Status -->
      <div class="card glass-card model-card">
        <h3 class="card-title">
          <el-icon><Box /></el-icon>
          基础模型
          <el-tag :type="modelStatus.exists ? 'success' : 'warning'" size="small" effect="dark" class="status-tag">
            {{ modelStatus.exists ? '就绪' : '需下载' }}
          </el-tag>
        </h3>
        
        <div class="model-info" v-if="modelStatus.summary">
          <div class="model-progress-ring">
            <svg viewBox="0 0 100 100">
              <circle class="bg" cx="50" cy="50" r="45" />
              <circle 
                class="progress" 
                cx="50" cy="50" r="45" 
                :style="{ strokeDashoffset: progressOffset }"
              />
            </svg>
            <div class="progress-text">
              <span class="number">{{ validPercent }}</span>
              <span class="percent">%</span>
            </div>
          </div>
          <div class="model-stats">
            <div class="stat">
              <span class="stat-value valid">{{ modelStatus.summary.valid_components }}</span>
              <span class="stat-label">有效组件</span>
            </div>
            <div class="stat">
              <span class="stat-value">{{ modelStatus.summary.total_components }}</span>
              <span class="stat-label">总组件</span>
            </div>
          </div>
        </div>

        <div class="component-list" v-if="modelStatus.details">
          <div 
            v-for="(comp, name) in modelStatus.details" 
            :key="name"
            class="component-item"
            :class="{ valid: comp.valid, missing: !comp.exists }"
          >
            <el-icon class="comp-icon">
              <CircleCheck v-if="comp.valid" />
              <Warning v-else-if="comp.exists" />
              <Close v-else />
            </el-icon>
            <span class="comp-name">{{ getComponentName(name) }}</span>
          </div>
        </div>

        <div class="model-actions" v-if="!modelStatus.exists">
          <el-button 
            v-if="!isDownloading" 
            type="primary" 
            @click="startDownload" 
            :loading="startingDownload"
          >
            <el-icon><Download /></el-icon>
            下载 Z-Image-Turbo 模型
          </el-button>
          <div v-else class="download-status">
            <el-progress :percentage="downloadProgress" :stroke-width="10" />
            <span class="download-text">{{ downloadSizeText }}</span>
          </div>
        </div>
      </div>

      <!-- About -->
      <div class="card glass-card about-card">
        <h3 class="card-title">
          <el-icon><InfoFilled /></el-icon>
          关于
        </h3>
        <div class="about-content">
          <p>
            <strong>None Trainer</strong> 是专为 Z-Image-Turbo 模型设计的 LoRA 训练工具，
            采用创新的 <em>AC-RF（锚点耦合整流流）</em> 采样策略，
            实现高效稳定的少步推理模型微调。
          </p>
          <div class="tech-highlights">
            <div class="highlight">
              <span class="highlight-icon">🎯</span>
              <span>锚点耦合采样</span>
            </div>
            <div class="highlight">
              <span class="highlight-icon">📉</span>
              <span>Min-SNR 加权</span>
            </div>
            <div class="highlight">
              <span class="highlight-icon">🔧</span>
              <span>自动硬件适配</span>
            </div>
            <div class="highlight">
              <span class="highlight-icon">⚡</span>
              <span>Flash Attention</span>
            </div>
          </div>
          <div class="author">
            <span>Made with ❤️ by <strong>None</strong></span>
          </div>
        </div>
      </div>
      
      <!-- Contact -->
      <div class="card glass-card contact-card">
        <h3 class="card-title">
          <el-icon><Message /></el-icon>
          合作交流
        </h3>
        <div class="contact-content">
          <p class="contact-intro">欢迎技术交流、问题反馈、商务合作</p>
          <div class="contact-list">
            <div class="contact-item">
              <span class="contact-icon">📧</span>
              <div class="contact-info">
                <span class="contact-label">Gmail</span>
                <span class="contact-value">lihaonan1082@gmail.com</span>
              </div>
              <el-button size="small" text @click="copyEmail('lihaonan1082@gmail.com')">
                <el-icon><CopyDocument /></el-icon>
              </el-button>
            </div>
            <div class="contact-item">
              <span class="contact-icon">📮</span>
              <div class="contact-info">
                <span class="contact-label">QQ邮箱</span>
                <span class="contact-value">592532681@qq.com</span>
              </div>
              <el-button size="small" text @click="copyEmail('592532681@qq.com')">
                <el-icon><CopyDocument /></el-icon>
              </el-button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, markRaw } from 'vue'
import { useSystemStore } from '@/stores/system'
import { useWebSocketStore } from '@/stores/websocket'
import { 
  Picture, Setting, VideoPlay, Cpu, Edit, Document, Clock, List, 
  CircleCheck, Warning, Close, Loading, Box, Monitor, Operation,
  ArrowRight, MagicStick, Download, InfoFilled, Lightning, TrendCharts,
  Message, CopyDocument
} from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const systemStore = useSystemStore()
const wsStore = useWebSocketStore()

const modelStatus = ref({ exists: false, details: null as any, summary: null as any })
const startingDownload = ref(false)

const systemInfo = computed(() => systemStore.systemInfo)
const wsConnected = computed(() => wsStore.isConnected)
const hasSystemInfo = computed(() => systemStore.systemInfo.python !== '')

const downloadStatus = computed(() => systemStore.downloadStatus)
const isDownloading = computed(() => downloadStatus.value.status === 'running')
const downloadProgress = computed(() => downloadStatus.value.progress)
const downloadSizeText = computed(() => {
  const gb = downloadStatus.value.downloaded_size_gb || 0
  return gb > 0 ? `已下载 ${gb.toFixed(2)} GB` : '准备中...'
})

const validPercent = computed(() => {
  if (!modelStatus.value.summary) return 0
  const { valid_components, total_components } = modelStatus.value.summary
  return Math.round((valid_components / total_components) * 100)
})

const progressOffset = computed(() => {
  const circumference = 2 * Math.PI * 45
  return circumference - (validPercent.value / 100) * circumference
})

const componentNames: Record<string, string> = {
  'transformer': 'Transformer',
  'vae': 'VAE',
  'text_encoder': 'Text Encoder',
  'tokenizer': 'Tokenizer',
  'scheduler': 'Scheduler',
  'model_index.json': 'Model Index'
}

function getComponentName(name: string): string {
  return componentNames[name] || name
}

async function refreshModelStatus() {
  try {
    const res = await axios.get('/api/system/model-status')
    modelStatus.value = res.data
  } catch (e) {
    console.error('Failed to check model status:', e)
  }
}

async function startDownload() {
  startingDownload.value = true
  try {
    await axios.post('/api/system/download-model')
    ElMessage.success('下载任务已启动')
  } catch (e: any) {
    ElMessage.error('启动下载失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    startingDownload.value = false
  }
}

refreshModelStatus()

function copyEmail(email: string) {
  navigator.clipboard.writeText(email)
  ElMessage.success(`已复制: ${email}`)
}
</script>

<style scoped>
.welcome-container {
  min-height: 100%;
  overflow-y: auto;
  background: var(--bg-primary);
}

/* Hero Section */
.hero-section {
  position: relative;
  padding: 60px 40px 40px;
  text-align: center;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(240, 180, 41, 0.03) 0%, transparent 100%);
}

.hero-content {
  position: relative;
  z-index: 1;
}

.logo-area {
  margin-bottom: 20px;
}

.logo-icon {
  width: 80px;
  height: 80px;
  margin: 0 auto;
  background: linear-gradient(135deg, #f0b429 0%, #e67e22 100%);
  border-radius: 20px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 10px 40px rgba(240, 180, 41, 0.3);
}

.logo-text {
  font-size: 42px;
  font-weight: 800;
  color: #1a1a1d;
  font-family: 'SF Pro Display', -apple-system, sans-serif;
}

.main-title {
  font-size: 3.5rem;
  font-weight: 800;
  margin: 0 0 12px 0;
  letter-spacing: -2px;
}

.title-gradient {
  background: linear-gradient(135deg, #f0b429 0%, #f39c12 50%, #e67e22 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.tagline {
  font-size: 1.3rem;
  color: var(--text-secondary);
  margin: 0 0 8px 0;
  font-weight: 500;
}

.description {
  font-size: 1rem;
  color: var(--text-muted);
  margin: 0 0 24px 0;
  max-width: 500px;
  margin-left: auto;
  margin-right: auto;
}

.description strong {
  color: var(--primary);
}

.feature-tags {
  display: flex;
  gap: 16px;
  justify-content: center;
  flex-wrap: wrap;
}

.feature-tags .tag {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 16px;
  background: rgba(240, 180, 41, 0.1);
  border: 1px solid rgba(240, 180, 41, 0.2);
  border-radius: 20px;
  font-size: 13px;
  color: var(--text-secondary);
}

.feature-tags .tag .el-icon {
  color: var(--primary);
}

/* Decoration */
.hero-decoration {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.deco-circle {
  position: absolute;
  border-radius: 50%;
  opacity: 0.5;
}

.deco-circle.c1 {
  width: 300px;
  height: 300px;
  top: -100px;
  right: -50px;
  background: radial-gradient(circle, rgba(240, 180, 41, 0.1) 0%, transparent 70%);
}

.deco-circle.c2 {
  width: 200px;
  height: 200px;
  bottom: 0;
  left: 10%;
  background: radial-gradient(circle, rgba(240, 180, 41, 0.08) 0%, transparent 70%);
}

.deco-circle.c3 {
  width: 150px;
  height: 150px;
  top: 30%;
  left: 5%;
  background: radial-gradient(circle, rgba(240, 180, 41, 0.06) 0%, transparent 70%);
}

/* Dashboard Grid */
.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 24px;
  padding: 0 40px 40px;
  max-width: 1200px;
  margin: 0 auto;
}

/* Cards */
.card {
  padding: 24px;
  border-radius: 16px;
  background: var(--bg-secondary);
  border: 1px solid var(--border-color);
}

.card-title {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 0 0 20px 0;
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-primary);
}

.card-title .el-icon {
  color: var(--primary);
}

.status-tag {
  margin-left: auto;
}

/* Quick Actions */
.quick-actions {
  grid-column: 1 / -1;
}

.action-buttons {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.action-item {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 16px;
  background: var(--bg-darker);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.action-item:hover {
  border-color: var(--primary);
  background: rgba(240, 180, 41, 0.05);
  transform: translateY(-2px);
}

.action-icon {
  width: 44px;
  height: 44px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}

.action-icon.dataset {
  background: rgba(64, 158, 255, 0.15);
  color: #409eff;
}

.action-icon.config {
  background: rgba(103, 194, 58, 0.15);
  color: #67c23a;
}

.action-icon.train {
  background: rgba(240, 180, 41, 0.15);
  color: #f0b429;
}

.action-icon.generate {
  background: rgba(230, 126, 34, 0.15);
  color: #e67e22;
}

.action-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.action-name {
  font-weight: 600;
  color: var(--text-primary);
}

.action-desc {
  font-size: 12px;
  color: var(--text-muted);
}

.action-item .arrow {
  color: var(--text-muted);
  transition: transform 0.2s;
}

.action-item:hover .arrow {
  color: var(--primary);
  transform: translateX(4px);
}

/* System Status */
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  padding: 10px 14px;
  background: var(--bg-darker);
  border-radius: 8px;
}

.status-item .label {
  color: var(--text-muted);
  font-size: 13px;
}

.status-item .value {
  color: var(--text-primary);
  font-weight: 500;
  font-size: 13px;
  font-family: var(--font-mono);
}

.loading-placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 40px;
  color: var(--text-muted);
}

/* Model Card */
.model-info {
  display: flex;
  align-items: center;
  gap: 24px;
  margin-bottom: 20px;
}

.model-progress-ring {
  position: relative;
  width: 100px;
  height: 100px;
  flex-shrink: 0;
}

.model-progress-ring svg {
  transform: rotate(-90deg);
}

.model-progress-ring circle {
  fill: none;
  stroke-width: 8;
  stroke-linecap: round;
}

.model-progress-ring .bg {
  stroke: var(--bg-darker);
}

.model-progress-ring .progress {
  stroke: var(--success);
  stroke-dasharray: 283;
  transition: stroke-dashoffset 0.5s ease;
}

.progress-text {
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  text-align: center;
}

.progress-text .number {
  font-size: 24px;
  font-weight: 700;
  color: var(--text-primary);
}

.progress-text .percent {
  font-size: 12px;
  color: var(--text-muted);
}

.model-stats {
  display: flex;
  gap: 24px;
}

.model-stats .stat {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.stat-value {
  font-size: 28px;
  font-weight: 700;
  color: var(--text-primary);
}

.stat-value.valid {
  color: var(--success);
}

.stat-label {
  font-size: 12px;
  color: var(--text-muted);
}

.component-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.component-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  background: var(--bg-darker);
  border-radius: 6px;
  font-size: 12px;
}

.component-item.valid {
  background: rgba(103, 194, 58, 0.1);
}

.component-item.valid .comp-icon {
  color: var(--success);
}

.component-item.missing {
  opacity: 0.5;
}

.component-item.missing .comp-icon {
  color: var(--text-muted);
}

.comp-name {
  color: var(--text-secondary);
}

.model-actions {
  text-align: center;
}

.download-status {
  max-width: 300px;
  margin: 0 auto;
}

.download-text {
  display: block;
  margin-top: 8px;
  font-size: 13px;
  color: var(--text-muted);
}

/* About Card */
.about-content p {
  color: var(--text-secondary);
  line-height: 1.7;
  margin: 0 0 16px 0;
}

.about-content strong {
  color: var(--primary);
}

.about-content em {
  color: var(--text-primary);
  font-style: normal;
  font-weight: 500;
}

.tech-highlights {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}

.highlight {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 14px;
  background: var(--bg-darker);
  border-radius: 8px;
  font-size: 13px;
  color: var(--text-secondary);
}

.highlight-icon {
  font-size: 16px;
}

/* Contact Card */
.contact-card {
  grid-column: 2;
}

.contact-intro {
  margin: 0 0 16px 0;
  color: var(--text-muted);
  font-size: 13px;
}

.contact-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.contact-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  background: var(--bg-darker);
  border-radius: 10px;
  border: 1px solid var(--border-color);
  transition: all 0.2s ease;
}

.contact-item:hover {
  border-color: var(--primary);
  background: rgba(240, 180, 41, 0.05);
}

.contact-icon {
  font-size: 24px;
}

.contact-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.contact-label {
  font-size: 11px;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.contact-value {
  font-size: 14px;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.contact-item .el-button {
  color: var(--text-muted);
}

.contact-item .el-button:hover {
  color: var(--primary);
}

.author {
  text-align: center;
  padding-top: 16px;
  color: var(--text-muted);
  font-size: 13px;
}

.author strong {
  color: var(--primary);
}

/* Responsive */
@media (max-width: 900px) {
  .dashboard-grid {
    grid-template-columns: 1fr;
    padding: 0 20px 20px;
  }
  
  .hero-section {
    padding: 40px 20px 30px;
  }
  
  .main-title {
    font-size: 2.5rem;
  }
  
  .action-buttons {
    grid-template-columns: 1fr;
  }
}
</style>
