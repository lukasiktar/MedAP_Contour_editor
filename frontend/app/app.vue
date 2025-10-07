<template>
  <div>
    <UApp>
      <NuxtRouteAnnouncer />
      <TheMenu />
      <NuxtPage />
    </UApp>
  </div>
</template>

<script setup lang="ts">
useSeoMeta({
  title: 'MedAP',
  ogTitle: 'MedAP',
  description: 'Help annotate data.',
  ogDescription: 'Help annotate data.',
  // ogImage: 'https://example.com/image.png',
  // twitterCard: 'summary_large_image',
})
</script>


<style>

.centered {
  display: flex;
  justify-content: center;
}

.centered-text {
  text-align: center;
}

</style>