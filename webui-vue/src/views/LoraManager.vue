<template>
  <div class="lora-manager">
    <div class="page-header">
      <h1><el-icon><Files /></el-icon> LoRA 模型管理</h1>
      <el-button @click="fetchLoras" :loading="loading">
        <el-icon><Refresh /></el-icon> 刷新
      </el-button>
    </div>

    <el-card class="lora-card glass-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span>训练产出 ({{ loraList.length }} 个模型)</span>
          <span class="path-hint">路径: {{ loraPath }}</span>
        </div>
      </template>

      <div v-loading="loading" class="lora-content">
        <el-empty v-if="loraList.length === 0 && !loading" description="暂无 LoRA 模型">
          <template #image>
            <el-icon style="font-size: 64px; color: var(--el-text-color-secondary)"><FolderOpened /></el-icon>
          </template>
        </el-empty>

        <el-table v-else :data="loraList" style="width: 100%" stripe>
          <el-table-column prop="name" label="文件名" min-width="300">
            <template #default="{ row }">
              <div class="file-name">
                <el-icon class="file-icon"><Document /></el-icon>
                <span>{{ row.name }}</span>
              </div>
            </template>
          </el-table-column>
          
          <el-table-column prop="size" label="大小" width="120" align="right">
            <template #default="{ row }">
              {{ formatSize(row.size) }}
            </template>
          </el-table-column>

          <el-table-column label="操作" width="200" align="center">
            <template #default="{ row }">
              <el-button-group>
                <el-button type="primary" size="small" @click="downloadLora(row)">
                  <el-icon><Download /></el-icon> 下载
                </el-button>
                <el-button type="danger" size="small" @click="deleteLora(row)">
                  <el-icon><Delete /></el-icon>
                </el-button>
              </el-button-group>
            </template>
          </el-table-column>
        </el-table>
      </div>
    </el-card>

    <!-- 删除确认对话框 -->
    <el-dialog v-model="deleteDialogVisible" title="确认删除" width="400px">
      <p>确定要删除 LoRA 模型吗？</p>
      <p class="delete-filename">{{ selectedLora?.name }}</p>
      <p class="warning-text">此操作不可恢复！</p>
      <template #footer>
        <el-button @click="deleteDialogVisible = false">取消</el-button>
        <el-button type="danger" @click="confirmDelete" :loading="deleting">删除</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Files, Refresh, Document, Download, Delete, FolderOpened } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

interface LoraItem {
  name: string
  path: string
  size: number
}

const loading = ref(false)
const loraList = ref<LoraItem[]>([])
const loraPath = ref('')
const deleteDialogVisible = ref(false)
const selectedLora = ref<LoraItem | null>(null)
const deleting = ref(false)

const formatSize = (bytes: number) => {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB'
}

const fetchLoras = async () => {
  loading.value = true
  try {
    const [lorasRes, pathsRes] = await Promise.all([
      axios.get('/api/loras'),
      axios.get('/api/training/system-paths')
    ])
    loraList.value = lorasRes.data
    loraPath.value = pathsRes.data.output_base_dir || './output'
  } catch (e) {
    console.error('Failed to fetch LoRAs:', e)
    ElMessage.error('获取 LoRA 列表失败')
  } finally {
    loading.value = false
  }
}

const downloadLora = (lora: LoraItem) => {
  // 直接用浏览器下载，不用 axios
  const link = document.createElement('a')
  link.href = `/api/loras/download?path=${encodeURIComponent(lora.path)}`
  link.setAttribute('download', lora.name.split('/').pop() || 'lora.safetensors')
  document.body.appendChild(link)
  link.click()
  link.remove()
  ElMessage.info('已开始下载')
}

const deleteLora = (lora: LoraItem) => {
  selectedLora.value = lora
  deleteDialogVisible.value = true
}

const confirmDelete = async () => {
  if (!selectedLora.value) return
  
  deleting.value = true
  try {
    await axios.delete(`/api/loras/delete?path=${encodeURIComponent(selectedLora.value.path)}`)
    ElMessage.success('删除成功')
    deleteDialogVisible.value = false
    fetchLoras()
  } catch (e) {
    console.error('Delete failed:', e)
    ElMessage.error('删除失败')
  } finally {
    deleting.value = false
  }
}

onMounted(() => {
  fetchLoras()
})
</script>

<style scoped>
.lora-manager {
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
}

.page-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.page-header h1 {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 24px;
  margin: 0;
}

.lora-card {
  background: var(--el-bg-color);
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.path-hint {
  font-size: 12px;
  color: var(--el-text-color-secondary);
  font-family: monospace;
}

.lora-content {
  min-height: 300px;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.file-icon {
  color: var(--el-color-primary);
}

.delete-filename {
  font-family: monospace;
  background: var(--el-fill-color-light);
  padding: 8px 12px;
  border-radius: 4px;
  word-break: break-all;
}

.warning-text {
  color: var(--el-color-danger);
  font-size: 12px;
}

:deep(.el-table) {
  --el-table-bg-color: transparent;
  --el-table-tr-bg-color: transparent;
}
</style>

