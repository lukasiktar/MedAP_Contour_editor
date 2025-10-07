<template>
    
    <div class="ann-container my-4 px-2">
        <div class="ann-img-container">
            <img src="/data/image.png" />
        </div>
        <div class="ann-ctrl-container">
            <div class="ann-ctrl-group">
                <UButton variant="outline" label="Upload dataset"></UButton>
            </div>
            <div class="ann-ctrl-group">
                <UButton v-if="sessionRunning" label="Stop" @click="toggleSession" color="error" variant="outline" />
                <UButton v-else label="Start" @click="toggleSession" />
            </div>
            <UButton>Save (Enter)</UButton>
            <UButton>Segment Image</UButton>
            <UButton>Empty Segmentation</UButton>
            <div class="ann-ctrl-group">
                <UButton>Add points</UButton>
                <UButton>Remove points</UButton>
            </div>
            <div class="ann-ctrl-group">
                <UButton>Zoom in</UButton>
                <UButton>Zoom out</UButton>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">

const sessionRunning = ref(false)

function toggleSession() {
    sessionRunning.value = !sessionRunning.value
}

function fetchImage() {
    // load image if possible
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