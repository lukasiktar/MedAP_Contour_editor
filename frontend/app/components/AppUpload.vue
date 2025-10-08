<template>
    <UModal>
        <UButton label="Upload data" variant="subtle" />

        <template #content>
            <!-- pick from media or upload new data -->
            <UFileUpload v-model="value" label="Drop images here." description="jpg, jpeg, png" />
        </template>
    </UModal>
</template>

<script setup lang="ts">

const value = ref(null)
</script>