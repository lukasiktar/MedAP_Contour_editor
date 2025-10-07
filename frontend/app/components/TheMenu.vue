<template>
    <UHeader title="MedAP">

        <UNavigationMenu :items="items" class="w-full justify-center" orientation="horizontal" />
        <UColorModeButton />

        <template #body>
            <UNavigationMenu :items="items" orientation="vertical" class="-mx-2.5" />
        </template>
    </UHeader>
    <!-- <USeparator label="MedAP" class="my-2" color="primary" /> -->
</template>

<script setup lang="ts">
import type { NavigationMenuItem } from '@nuxt/ui'

const items = ref<NavigationMenuItem[]>([
    {
        label: 'Home',
        to: '/'
    },
    {
        label: 'Settings',
        to: '/settings'
    },
    {
        label: 'Help',
        to: '/help'
    }
])
</script>

<style scoped></style>