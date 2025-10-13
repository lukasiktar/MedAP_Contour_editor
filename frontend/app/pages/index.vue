<template>

    <div class="ann-container my-4 px-1">

        <ClientOnly>
            <div class="ann-img-container">
                <v-stage :config="stageConfig" @mousedown="handleMouseDown" @mousemove="handleMouseMove"
                    @mouseup="handleMouseUp" @touchstart="handleMouseDown" @touchmove="handleMouseMove"
                    @touchend="handleMouseUp">
                    <v-layer ref="layerRef">
                        <v-image ref="backgroundRef" :config="backgroundConfig" />
                        <v-image ref="imageRef" :config="imageConfig" />
                    </v-layer>
                </v-stage>
            </div>
        </ClientOnly>

        <div class="ann-ctrl-container">
            <div class="ann-ctrl-group">
                <p>#1/100</p>
            </div>
            <div class="ann-ctrl-group">
                <AppUpload />
            </div>
            <div class="ann-ctrl-group">
                <UButton v-if="sessionRunning" label="Stop" @click="toggleSession" color="error" variant="outline" />
                <UButton v-else label="Start" @click="toggleSession" />
            </div>
            <div class="ann-ctrl-group">
                <UButton>Clear (C)</UButton>
                <UButton>Save (Enter)</UButton>
            </div>
            <UButton>Segment Image (S)</UButton>
            <UButton>Empty Segmentation (E)</UButton>
            <div class="ann-ctrl-group">
                <UButton>Zoom in</UButton>
                <UButton>Zoom out</UButton>
            </div>
            <!-- unnecessary as current user will be marked automatically -->
            <div class="ann-ctrl-group">
                <UInputMenu v-model="value" :items="items" />
            </div>
        </div>

    </div>
</template>

<script setup lang="ts">

const sessionRunning = ref(false)

function toggleSession() {
    sessionRunning.value = !sessionRunning.value
}

const backgroundRef = ref(null)
const backgroundImage = ref<HTMLImageElement | null>(null)
const backgroundConfig = ref({ image: null as HTMLImageElement | null, x: 0, y: 0, scaleX: 1, scaleY: 1 })

const items = ref(['Ann1', 'Ann2', 'Ann3', 'Ann4'])
const value = ref('Ann1')

// bruhsing
const tool = ref('brush');
const isDrawing = ref(false);
const lastPos = ref(null);
const imageRef = ref(null);
const layerRef = ref(null);

const stageConfig = ref({ width: 0, height: 0 })
const canvas = ref<HTMLCanvasElement | null>(null)
const context = ref<CanvasRenderingContext2D | null>(null)
const imageConfig = ref({ image: null as HTMLCanvasElement | null, x: 0, y: 0 })

const loadBackground = (path: string) => {
    // prevent running on server
    // if (process.server) return

    const img = new Image()

    // attach handler before setting src -> prevents race conditions
    img.addEventListener('load', () => {
        backgroundImage.value = img

        const width = stageConfig.value.width
        const height = stageConfig.value.height

        const scale = Math.min(width / img.width, height / img.height)

        backgroundConfig.value = {
            image: img,
            x: 0,
            y: 0,
            scaleX: scale,
            scaleY: scale,
        }

        // redraw konva layer if available
        // layerRef.value?.getNode()?.batchDraw()
    })

    img.addEventListener('error', (err) => {
        console.error('Failed to load image', err)
    })

    img.src = path
}

onMounted(() => {
    const width = window.innerWidth * 0.85
    const height = window.innerHeight * 0.9

    stageConfig.value = { width, height }

    // now safe: we are in the browser
    canvas.value = document.createElement('canvas')
    canvas.value.width = width
    canvas.value.height = height

    const ctx = canvas.value.getContext('2d')
    if (!ctx) return

    ctx.strokeStyle = '#df4b26'
    ctx.lineJoin = 'round'
    ctx.lineWidth = 5

    context.value = ctx
    imageConfig.value.image = canvas.value

    loadBackground('/data/image.png')
})


const handleMouseDown = (e) => {
    isDrawing.value = true;
    console.log('draw')
    lastPos.value = e.target.getStage().getPointerPosition();
};

const handleMouseUp = () => {
    isDrawing.value = false;
};

const handleMouseMove = (e) => {
    if (!isDrawing.value || !context.value || !canvas.value) return

    const ctx = context.value
    const image = imageRef.value.getNode()
    const stage = e.target.getStage()

    ctx.globalCompositeOperation =
        tool.value === 'eraser' ? 'destination-out' : 'source-over'
    ctx.beginPath()

    const localPos = {
        x: lastPos.value.x - image.x(),
        y: lastPos.value.y - image.y(),
    }
    ctx.moveTo(localPos.x, localPos.y)

    const pos = stage.getPointerPosition()
    const newLocalPos = {
        x: pos.x - image.x(),
        y: pos.y - image.y(),
    }
    ctx.lineTo(newLocalPos.x, newLocalPos.y)
    ctx.closePath()
    ctx.stroke()

    lastPos.value = pos
    layerRef.value.getNode().batchDraw()
}

const eraseAnnotation = () => {
    // TODO
    layerRef = null;
}

</script>

<style scoped>
.ann-container {
    max-height: 100vh;
    width: 100vw;
    display: flex;
    flex-direction: row;
    align-items: stretch;
    justify-content: space-around;
    align-items: center;
    overflow: hidden;
}

.ann-img-container {
    height: 90vh;
    flex: 0 0 85%;
    display: flex;
}

.ann-img img {
    max-height: 100%;
    max-width: 100%;
    height: auto;
    width: auto;
    object-fit: contain;
    display: block;
}

.ann-ctrl-container {
    flex: 0 0 10%;
    display: flex;
    flex-direction: column;
    gap: 1em;
    align-items: center;
}

.ann-ctrl-group {
    display: flex;
    flex-direction: row;
    gap: 1em;
}
</style>