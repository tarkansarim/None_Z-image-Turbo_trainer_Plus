<template>
  <div class="job-history">
    <div class="page-header">
      <h1 class="gradient-text">
        <el-icon><Clock /></el-icon> 训练历史
      </h1>
      <div class="header-actions">
        <el-button @click="refreshJobs" :loading="jobsStore.loading">
          <el-icon><Refresh /></el-icon> 刷新
        </el-button>
      </div>
    </div>

    <!-- Filters -->
    <el-card class="filter-card glass-card" shadow="never">
      <div class="filter-row">
        <el-select 
          v-model="statusFilter" 
          placeholder="状态筛选" 
          @change="handleFilterChange"
          style="width: 140px"
        >
          <el-option label="全部状态" value="all" />
          <el-option label="待开始" value="pending" />
          <el-option label="运行中" value="running" />
          <el-option label="已暂停" value="stopped" />
          <el-option label="已完成" value="completed" />
          <el-option label="失败" value="failed" />
        </el-select>
        
        <el-input 
          v-model="searchQuery" 
          placeholder="搜索任务名称..." 
          clearable
          @clear="handleFilterChange"
          @keyup.enter="handleFilterChange"
          style="width: 240px"
        >
          <template #prefix>
            <el-icon><Search /></el-icon>
          </template>
        </el-input>
        
        <span class="result-count">
          共 <strong>{{ jobsStore.totalJobs }}</strong> 个任务
        </span>
      </div>
    </el-card>

    <!-- Jobs Table -->
    <el-card class="jobs-card glass-card" shadow="never">
      <div v-loading="jobsStore.loading" class="jobs-content">
        <el-empty v-if="jobsStore.jobs.length === 0 && !jobsStore.loading" description="暂无训练任务">
          <template #image>
            <el-icon style="font-size: 64px; color: var(--el-text-color-secondary)"><DocumentRemove /></el-icon>
          </template>
          <el-button type="primary" @click="$router.push('/config')">
            <el-icon><Plus /></el-icon> 创建训练配置
          </el-button>
        </el-empty>

        <el-table 
          v-else 
          :data="jobsStore.jobs" 
          style="width: 100%" 
          stripe
          row-key="id"
        >
          <el-table-column prop="name" label="任务名称" min-width="200">
            <template #default="{ row }">
              <div class="job-name-cell">
                <el-icon class="job-icon" :style="{ color: jobsStore.getStatusColor(row.status) }">
                  <Loading v-if="row.status === 'running'" class="spin" />
                  <CircleCheck v-else-if="row.status === 'completed'" />
                  <WarningFilled v-else-if="row.status === 'failed'" />
                  <VideoPause v-else-if="row.status === 'stopped'" />
                  <Clock v-else />
                </el-icon>
                <span class="job-name">{{ row.name }}</span>
              </div>
            </template>
          </el-table-column>

          <el-table-column prop="status" label="状态" width="100" align="center">
            <template #default="{ row }">
              <el-tag 
                :type="getTagType(row.status)" 
                size="small"
                :effect="row.status === 'running' ? 'dark' : 'plain'"
              >
                {{ jobsStore.getStatusLabel(row.status) }}
              </el-tag>
            </template>
          </el-table-column>

          <el-table-column label="进度" width="180" align="center">
            <template #default="{ row }">
              <div class="progress-cell">
                <el-progress 
                  :percentage="jobsStore.getProgressPercent(row)" 
                  :stroke-width="8"
                  :status="row.status === 'completed' ? 'success' : row.status === 'failed' ? 'exception' : undefined"
                />
                <span class="progress-text">
                  {{ row.current_step || 0 }}/{{ row.total_steps || '?' }}
                </span>
              </div>
            </template>
          </el-table-column>

          <el-table-column prop="final_loss" label="Loss" width="100" align="right">
            <template #default="{ row }">
              <span class="loss-value" v-if="row.final_loss !== null">
                {{ row.final_loss.toFixed(4) }}
              </span>
              <span v-else class="text-muted">-</span>
            </template>
          </el-table-column>

          <el-table-column label="时间" width="140" align="center">
            <template #default="{ row }">
              <div class="time-cell">
                <span class="time-label">创建:</span>
                <span>{{ jobsStore.formatDate(row.created_at) }}</span>
              </div>
            </template>
          </el-table-column>

          <el-table-column label="操作" width="280" align="center">
            <template #default="{ row }">
              <div class="action-buttons">
                <!-- Resume button -->
                <el-tooltip content="继续训练" v-if="canResume(row)">
                  <el-button 
                    type="primary" 
                    size="small" 
                    @click="handleResume(row)"
                    :disabled="jobsStore.hasRunningJob && row.status !== 'running'"
                  >
                    <el-icon><VideoPlay /></el-icon>
                  </el-button>
                </el-tooltip>
                
                <!-- Stop button -->
                <el-tooltip content="停止训练" v-if="row.status === 'running'">
                  <el-button type="warning" size="small" @click="handleStop(row)">
                    <el-icon><VideoPause /></el-icon>
                  </el-button>
                </el-tooltip>
                
                <!-- Edit button -->
                <el-tooltip content="编辑配置" v-if="row.status !== 'running'">
                  <el-button size="small" @click="handleEdit(row)">
                    <el-icon><Edit /></el-icon>
                  </el-button>
                </el-tooltip>
                
                <!-- View logs button -->
                <el-tooltip content="查看日志">
                  <el-button size="small" @click="handleViewLogs(row)">
                    <el-icon><Document /></el-icon>
                  </el-button>
                </el-tooltip>
                
                <!-- Duplicate button -->
                <el-tooltip content="复制任务">
                  <el-button size="small" @click="handleDuplicate(row)">
                    <el-icon><CopyDocument /></el-icon>
                  </el-button>
                </el-tooltip>
                
                <!-- Delete button -->
                <el-tooltip content="删除任务" v-if="row.status !== 'running'">
                  <el-button type="danger" size="small" @click="handleDelete(row)">
                    <el-icon><Delete /></el-icon>
                  </el-button>
                </el-tooltip>
              </div>
            </template>
          </el-table-column>
        </el-table>

        <!-- Pagination -->
        <div class="pagination-wrapper" v-if="jobsStore.totalJobs > jobsStore.pageSize">
          <el-pagination
            v-model:current-page="currentPage"
            :page-size="jobsStore.pageSize"
            :total="jobsStore.totalJobs"
            layout="prev, pager, next"
            @current-change="handlePageChange"
          />
        </div>
      </div>
    </el-card>

    <!-- Logs Dialog -->
    <el-dialog v-model="logsDialogVisible" :title="`日志 - ${selectedJob?.name || ''}`" width="800px" top="5vh">
      <div class="logs-container">
        <div class="loss-chart-container" v-if="lossHistory && lossHistory.steps.length > 0">
          <h4>Loss 曲线</h4>
          <div class="mini-chart">
            <canvas ref="lossChartCanvas"></canvas>
          </div>
        </div>
        
        <h4>训练日志</h4>
        <div class="logs-scroll" v-loading="logsLoading">
          <div v-if="jobLogs.length === 0" class="empty-logs">暂无日志</div>
          <div 
            v-for="(log, index) in jobLogs" 
            :key="index" 
            class="log-entry"
            :class="log.level"
          >
            <span class="log-time">{{ formatLogTime(log.timestamp) }}</span>
            <span class="log-level" :class="log.level">{{ log.level.toUpperCase() }}</span>
            <span class="log-message">{{ log.message }}</span>
          </div>
        </div>
      </div>
    </el-dialog>

    <!-- Edit Config Dialog -->
    <el-dialog v-model="editDialogVisible" title="编辑训练配置" width="600px" top="5vh">
      <div v-if="editConfig" class="edit-config-form">
        <el-form label-width="120px">
          <el-form-item label="任务名称">
            <el-input v-model="editConfig.training.output_name" />
          </el-form-item>
          
          <el-divider content-position="left">训练参数</el-divider>
          
          <el-form-item label="训练轮数">
            <el-input-number v-model="editConfig.advanced.num_train_epochs" :min="1" :max="1000" />
          </el-form-item>
          
          <el-form-item label="学习率">
            <el-input v-model="editConfig.training.learning_rate" />
          </el-form-item>
          
          <el-form-item label="批次大小">
            <el-input-number v-model="editConfig.dataset.batch_size" :min="1" :max="64" />
          </el-form-item>
          
          <el-form-item label="梯度累积">
            <el-input-number v-model="editConfig.advanced.gradient_accumulation_steps" :min="1" :max="128" />
          </el-form-item>
          
          <el-divider content-position="left">LoRA 参数</el-divider>
          
          <el-form-item label="Network Dim">
            <el-input-number v-model="editConfig.network.dim" :min="1" :max="512" />
          </el-form-item>
          
          <el-form-item label="Network Alpha">
            <el-input-number v-model="editConfig.network.alpha" :min="1" :max="512" />
          </el-form-item>
        </el-form>
      </div>
      <template #footer>
        <el-button @click="editDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="saveEditConfig" :loading="saving">保存</el-button>
      </template>
    </el-dialog>

    <!-- Delete Confirmation Dialog -->
    <el-dialog v-model="deleteDialogVisible" title="确认删除" width="400px">
      <p>确定要删除任务 <strong>{{ selectedJob?.name }}</strong> 吗？</p>
      <el-checkbox v-model="deleteOutputs" label="同时删除输出文件（checkpoint 和 LoRA）" />
      <p class="warning-text">⚠️ 此操作不可恢复！</p>
      <template #footer>
        <el-button @click="deleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="confirmDelete" :loading="deleting">删除</el-button>
      </template>
    </el-dialog>

    <!-- Duplicate Dialog -->
    <el-dialog v-model="duplicateDialogVisible" title="复制任务" width="400px">
      <el-form>
        <el-form-item label="新任务名称">
          <el-input v-model="duplicateName" placeholder="留空则自动添加 _copy 后缀" />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="duplicateDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="confirmDuplicate" :loading="duplicating">复制</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue'
import { useRouter } from 'vue-router'
import { 
  Clock, Refresh, Search, Plus, Edit, Delete, Document, CopyDocument,
  VideoPlay, VideoPause, CircleCheck, WarningFilled, Loading, DocumentRemove
} from '@element-plus/icons-vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { useJobsStore, type Job, type JobLog, type JobLossHistory } from '@/stores/jobs'
import Chart from 'chart.js/auto'

const router = useRouter()
const jobsStore = useJobsStore()

// Filters
const statusFilter = ref('all')
const searchQuery = ref('')
const currentPage = ref(1)

// Dialogs
const logsDialogVisible = ref(false)
const editDialogVisible = ref(false)
const deleteDialogVisible = ref(false)
const duplicateDialogVisible = ref(false)

// State
const selectedJob = ref<Job | null>(null)
const jobLogs = ref<JobLog[]>([])
const lossHistory = ref<JobLossHistory | null>(null)
const logsLoading = ref(false)
const editConfig = ref<Record<string, any> | null>(null)
const saving = ref(false)
const deleting = ref(false)
const duplicating = ref(false)
const deleteOutputs = ref(false)
const duplicateName = ref('')

// Chart
const lossChartCanvas = ref<HTMLCanvasElement | null>(null)
let chartInstance: Chart | null = null

function getTagType(status: string): '' | 'success' | 'warning' | 'danger' | 'info' {
  const types: Record<string, '' | 'success' | 'warning' | 'danger' | 'info'> = {
    pending: 'info',
    running: '',
    stopped: 'warning',
    completed: 'success',
    failed: 'danger'
  }
  return types[status] || 'info'
}

function canResume(job: Job): boolean {
  return ['pending', 'stopped', 'failed'].includes(job.status)
}

function formatLogTime(timestamp: string): string {
  if (!timestamp) return ''
  const date = new Date(timestamp)
  return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
}

async function refreshJobs() {
  await jobsStore.fetchJobs({
    status: statusFilter.value,
    search: searchQuery.value
  })
  await jobsStore.fetchRunningJob()
}

function handleFilterChange() {
  currentPage.value = 1
  jobsStore.fetchJobs({
    status: statusFilter.value,
    search: searchQuery.value
  })
}

function handlePageChange(page: number) {
  currentPage.value = page
  jobsStore.setPage(page)
}

async function handleResume(job: Job) {
  try {
    await jobsStore.resumeJob(job.id)
    ElMessage.success(`任务 ${job.name} 已开始`)
    router.push('/training')
  } catch (err: any) {
    ElMessage.error(err.response?.data?.detail || '启动任务失败')
  }
}

async function handleStop(job: Job) {
  try {
    await jobsStore.stopJob(job.id)
    ElMessage.warning(`任务 ${job.name} 已停止`)
    refreshJobs()
  } catch (err: any) {
    ElMessage.error(err.response?.data?.detail || '停止任务失败')
  }
}

async function handleViewLogs(job: Job) {
  selectedJob.value = job
  logsDialogVisible.value = true
  logsLoading.value = true
  
  try {
    const [logs, history] = await Promise.all([
      jobsStore.fetchJobLogs(job.id),
      jobsStore.fetchJobLossHistory(job.id)
    ])
    jobLogs.value = logs
    lossHistory.value = history
    
    // Draw chart after dialog is shown
    if (history && history.steps.length > 0) {
      await nextTick()
      drawLossChart(history)
    }
  } catch (err) {
    console.error('Failed to fetch logs:', err)
  } finally {
    logsLoading.value = false
  }
}

function drawLossChart(history: JobLossHistory) {
  if (!lossChartCanvas.value) return
  
  if (chartInstance) {
    chartInstance.destroy()
  }
  
  const ctx = lossChartCanvas.value.getContext('2d')
  if (!ctx) return
  
  chartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: history.steps,
      datasets: [{
        label: 'Loss',
        data: history.loss,
        borderColor: '#e8c47c',
        backgroundColor: 'rgba(232, 196, 124, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 0
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: {
          display: true,
          title: { display: true, text: 'Step', color: '#a0a0a8' },
          ticks: { color: '#6b6b73' },
          grid: { color: 'rgba(255,255,255,0.06)' }
        },
        y: {
          display: true,
          title: { display: true, text: 'Loss', color: '#a0a0a8' },
          ticks: { color: '#6b6b73' },
          grid: { color: 'rgba(255,255,255,0.06)' }
        }
      }
    }
  })
}

async function handleEdit(job: Job) {
  selectedJob.value = job
  
  // Fetch full job details to get config snapshot
  const fullJob = await jobsStore.fetchJob(job.id)
  if (fullJob && fullJob.config_snapshot) {
    editConfig.value = JSON.parse(JSON.stringify(fullJob.config_snapshot))
    editDialogVisible.value = true
  } else {
    ElMessage.error('无法加载任务配置')
  }
}

async function saveEditConfig() {
  if (!selectedJob.value || !editConfig.value) return
  
  saving.value = true
  try {
    await jobsStore.updateJobConfig(selectedJob.value.id, editConfig.value)
    ElMessage.success('配置已保存')
    editDialogVisible.value = false
    refreshJobs()
  } catch (err: any) {
    ElMessage.error(err.response?.data?.detail || '保存失败')
  } finally {
    saving.value = false
  }
}

function handleDelete(job: Job) {
  selectedJob.value = job
  deleteOutputs.value = false
  deleteDialogVisible.value = true
}

async function confirmDelete() {
  if (!selectedJob.value) return
  
  deleting.value = true
  try {
    await jobsStore.deleteJob(selectedJob.value.id, deleteOutputs.value)
    ElMessage.success('任务已删除')
    deleteDialogVisible.value = false
    refreshJobs()
  } catch (err: any) {
    ElMessage.error(err.response?.data?.detail || '删除失败')
  } finally {
    deleting.value = false
  }
}

function handleDuplicate(job: Job) {
  selectedJob.value = job
  duplicateName.value = ''
  duplicateDialogVisible.value = true
}

async function confirmDuplicate() {
  if (!selectedJob.value) return
  
  duplicating.value = true
  try {
    const newJobId = await jobsStore.duplicateJob(selectedJob.value.id, duplicateName.value || undefined)
    if (newJobId) {
      ElMessage.success('任务已复制')
      duplicateDialogVisible.value = false
      refreshJobs()
    }
  } catch (err: any) {
    ElMessage.error(err.response?.data?.detail || '复制失败')
  } finally {
    duplicating.value = false
  }
}

// Cleanup chart on dialog close
watch(logsDialogVisible, (visible) => {
  if (!visible && chartInstance) {
    chartInstance.destroy()
    chartInstance = null
  }
})

onMounted(() => {
  refreshJobs()
})
</script>

<style scoped lang="scss">
.job-history {
  padding: var(--space-lg);
  max-width: 1400px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: var(--space-lg);
  
  h1 {
    display: flex;
    align-items: center;
    gap: var(--space-sm);
    font-size: 1.75rem;
    margin: 0;
  }
}

.filter-card {
  margin-bottom: var(--space-md);
  
  :deep(.el-card__body) {
    padding: var(--space-md);
  }
}

.filter-row {
  display: flex;
  align-items: center;
  gap: var(--space-md);
}

.result-count {
  margin-left: auto;
  color: var(--text-secondary);
  font-size: 0.9rem;
  
  strong {
    color: var(--primary);
  }
}

.jobs-card {
  :deep(.el-card__body) {
    padding: 0;
  }
}

.jobs-content {
  min-height: 400px;
}

.job-name-cell {
  display: flex;
  align-items: center;
  gap: var(--space-sm);
  
  .job-icon {
    font-size: 18px;
  }
  
  .job-name {
    font-weight: 500;
  }
}

.spin {
  animation: spin 1s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.progress-cell {
  display: flex;
  flex-direction: column;
  gap: 4px;
  
  .progress-text {
    font-size: 0.75rem;
    color: var(--text-muted);
  }
}

.loss-value {
  font-family: var(--font-mono);
  color: var(--primary);
}

.text-muted {
  color: var(--text-muted);
}

.time-cell {
  font-size: 0.85rem;
  color: var(--text-secondary);
  
  .time-label {
    color: var(--text-muted);
    margin-right: 4px;
  }
}

.action-buttons {
  display: flex;
  gap: 4px;
  justify-content: center;
  flex-wrap: wrap;
}

.pagination-wrapper {
  display: flex;
  justify-content: center;
  padding: var(--space-md);
  border-top: 1px solid var(--border);
}

// Logs Dialog
.logs-container {
  max-height: 70vh;
  overflow-y: auto;
  
  h4 {
    margin: var(--space-md) 0 var(--space-sm);
    color: var(--text-secondary);
    
    &:first-child {
      margin-top: 0;
    }
  }
}

.loss-chart-container {
  margin-bottom: var(--space-lg);
  
  .mini-chart {
    height: 200px;
    background: var(--bg-input);
    border-radius: var(--radius-md);
    padding: var(--space-sm);
  }
}

.logs-scroll {
  max-height: 400px;
  overflow-y: auto;
  background: var(--bg-input);
  border-radius: var(--radius-md);
  padding: var(--space-sm);
}

.empty-logs {
  text-align: center;
  color: var(--text-muted);
  padding: var(--space-lg);
}

.log-entry {
  display: flex;
  gap: var(--space-sm);
  padding: 4px var(--space-sm);
  font-family: var(--font-mono);
  font-size: 0.85rem;
  border-radius: var(--radius-sm);
  
  &:hover {
    background: var(--bg-hover);
  }
  
  &.error {
    background: rgba(245, 108, 108, 0.1);
  }
  
  &.warning {
    background: rgba(230, 162, 60, 0.1);
  }
  
  &.success {
    background: rgba(103, 194, 58, 0.1);
  }
}

.log-time {
  color: var(--text-muted);
  white-space: nowrap;
}

.log-level {
  font-weight: 600;
  white-space: nowrap;
  min-width: 50px;
  
  &.info { color: var(--info); }
  &.error { color: var(--error); }
  &.warning { color: var(--warning); }
  &.success { color: var(--success); }
}

.log-message {
  color: var(--text-primary);
  word-break: break-all;
}

// Edit Config Dialog
.edit-config-form {
  max-height: 60vh;
  overflow-y: auto;
}

// Delete Dialog
.warning-text {
  color: var(--error);
  font-size: 0.9rem;
  margin-top: var(--space-md);
}
</style>

